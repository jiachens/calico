import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

'''
steal from existing open clip implementation
'''

from mmdet.models.builder import LOSSES

try:
    import torch.distributed.nn
    from torch import distributed as dist2
    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

def get_world_size():
    if not dist2.is_available():
        return 1
    if not dist2.is_initialized():
        return 1
    return dist2.get_world_size()


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to("cuda")
    size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
    dist2.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist2.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list

def gather_features(
        image_features,
        text_features,
        # local_loss=False,
        # gather_with_grad=False,
        # rank=0,
        # world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        all_image_features = hvd.allgather(image_features)
        all_text_features = hvd.allgather(text_features)

    else:#torch.distributed.nn.
        all_image_features = torch.cat(all_gather(image_features), dim=0)
        all_text_features = torch.cat(all_gather(text_features), dim=0)

    return all_image_features, all_text_features

@LOSSES.register_module()
class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            batch_loss=[False,-1],
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.use_horovod = use_horovod
        self.batch_loss, self.batch_size = batch_loss[0], batch_loss[1]
        # cache state
        # self.prev_num_logits = 0
        # self.labels = {}

    def forward(self, image_features, text_features, logit_scale):
        device = image_features.device

        if self.batch_loss:
            image_features = image_features.reshape(self.batch_size, -1, image_features.shape[-1])
            text_features = text_features.reshape(self.batch_size, -1, text_features.shape[-1])
            logits_image = logit_scale * torch.bmm(image_features, text_features.permute(0, 2, 1))
            logits_text =  logit_scale * torch.bmm(text_features, image_features.permute(0, 2, 1))

        elif self.local_loss:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        else:
            all_image_features, all_text_features = gather_features(
                image_features, text_features, self.use_horovod)
            logits_per_image = logit_scale * all_image_features @ all_text_features.T
            logits_per_text = logits_per_image.T

        if self.batch_loss:
            total_loss = 0
            num_logits = logits_image.shape[1]
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            for i in range(self.batch_size):
                total_loss += (
                    F.cross_entropy(logits_image[i], labels) +
                    F.cross_entropy(logits_text[i], labels)
                ) / 2
            total_loss /= self.batch_size
        else:
            num_logits = logits_per_image.shape[0]

            labels = torch.arange(num_logits, device=device, dtype=torch.long)

            total_loss = (
                F.cross_entropy(logits_per_image, labels) +
                F.cross_entropy(logits_per_text, labels)
                ) / 2
        return total_loss