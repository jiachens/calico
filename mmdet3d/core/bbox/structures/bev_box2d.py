import numpy as np
import torch


class BEVBox2D:
    def __init__(self,tensor):
        if isinstance(tensor, torch.Tensor):
            device = tensor.device
        else:
            device = torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that
            # does not depend on the inputs (and consequently confuses jit)
            tensor = tensor.reshape((0, 4)).to(
                dtype=torch.float32, device=device
            )
        assert tensor.dim() == 2 and tensor.size(-1) == 4, tensor.size()

        self.tensor = tensor

    @property
    def corners(self):
        corner_left_bottem = self.tensor[:,:2]
        corner_left_top = torch.stack([self.tensor[:,0], self.tensor[:,3]], dim=1)
        corner_right_bottem = torch.stack([self.tensor[:,2], self.tensor[:,1]], dim=1) 
        corner_right_top = self.tensor[:,2:]
        return corner_left_bottem, corner_left_top, corner_right_bottem, corner_right_top

    def flip(self, bev_direction="horizontal",):
        assert bev_direction in ("horizontal", "vertical")
        if bev_direction == "horizontal":
            self.tensor[:, 1] = -self.tensor[:, 3]
            self.tensor[:, 3] = -self.tensor[:, 1]
        elif bev_direction == "vertical":
            self.tensor[:, 0] = -self.tensor[:, 2]
            self.tensor[:, 2] = -self.tensor[:, 0]

    def rotate(self, angle):
        #TODO: add support for angle with [2,2] matrix
        if not isinstance(angle, torch.Tensor):
            angle = self.tensor.new_tensor(angle)
        assert (
            angle.numel() == 1
        ), f"invalid rotation angle shape {angle.shape}, we only support single angle rotation for now"

        rot_sin = torch.sin(angle)
        rot_cos = torch.cos(angle)
        rot_mat_T = self.tensor.new_tensor(
            [[rot_cos, -rot_sin], [rot_sin, rot_cos]]
        )
        corner_left_bottem, corner_left_top, corner_right_bottem, corner_right_top = self.corners
        corner_left_bottem = corner_left_bottem @ rot_mat_T
        corner_left_top = corner_left_top @ rot_mat_T
        corner_right_bottem = corner_right_bottem @ rot_mat_T
        corner_right_top = corner_right_top @ rot_mat_T

        self.tensor[:,0] = torch.minimum(corner_left_bottem[:,0], corner_right_bottem[:,0], corner_left_top[:,0], corner_right_top[:,0])
        self.tensor[:,1] = torch.minimum(corner_left_bottem[:,1], corner_right_bottem[:,1], corner_left_top[:,1], corner_right_top[:,1])
        self.tensor[:,2] = torch.maximum(corner_left_bottem[:,0], corner_right_bottem[:,0], corner_left_top[:,0], corner_right_top[:,0])
        self.tensor[:,3] = torch.maximum(corner_left_bottem[:,1], corner_right_bottem[:,1], corner_left_top[:,1], corner_right_top[:,1])
        # return rot_mat_T

    def scale(self,scale_factor):
        self.tensor *= scale_factor

    def translate(self, trans_vector):
        """Translate points with the given translation vector.
        """
        if not isinstance(trans_vector, torch.Tensor):
            trans_vector = self.tensor.new_tensor(trans_vector)
        trans_vector = trans_vector.squeeze(0)
        if trans_vector.dim() == 1:
            assert trans_vector.shape[0] == 3
            self.tensor[:, :2] += trans_vector[:2]
            self.tensor[:, 2:] += trans_vector[:2]
        elif trans_vector.dim() == 2:
            assert (
                trans_vector.shape[0] == self.tensor.shape[0]
                and trans_vector.shape[1] == 3
            )
            self.tensor[:, :2] += trans_vector[:, :2]
            self.tensor[:, 2:] += trans_vector[:, :2]
        else:
            raise NotImplementedError(
                f"Unsupported translation vector of shape {trans_vector.shape}"
            )

    def in_range_bev(self, box_range):
        in_range_flags = (
            (self.tensor[:, 0] > box_range[0])
            & (self.tensor[:, 1] > box_range[1])
            & (self.tensor[:, 2] < box_range[2])
            & (self.tensor[:, 3] < box_range[3])
        )
        return in_range_flags

    def __getitem__(self, item):
        """
        THE FOLLOWING COMMENTS ARE FROM POINTS CLASS
        Note:
            The following usage are allowed:
            1. `new_points = points[3]`:
                return a `Points` that contains only one point.
            2. `new_points = points[2:10]`:
                return a slice of points.
            3. `new_points = points[vector]`:
                where vector is a torch.BoolTensor with `length = len(points)`.
                Nonzero elements in the vector will be selected.
            Note that the returned Points might share storage with this Points,
            subject to Pytorch's indexing semantics.

        Returns:
            :obj:`BasePoints`: A new object of  \
                :class:`BasePoints` after indexing.
        """
        original_type = type(self)
        if isinstance(item, int):
            return original_type(
                self.tensor[item].view(1, -1)
            )
        elif isinstance(item, tuple) and len(item) == 2:
            if isinstance(item[1], slice):
                start = 0 if item[1].start is None else item[1].start
                stop = self.tensor.shape[1] if item[1].stop is None else item[1].stop
                step = 1 if item[1].step is None else item[1].step
                item = list(item)
                item[1] = list(range(start, stop, step))
                item = tuple(item)
            elif isinstance(item[1], int):
                item = list(item)
                item[1] = [item[1]]
                item = tuple(item)
            p = self.tensor[item[0], item[1]]
        else:
            raise NotImplementedError(f"Invalid slice {item}!")

        assert (
            p.dim() == 2
        ), f"Indexing on 2D BBox with {item} failed to return a matrix!"
        return original_type(p)

    def __len__(self):
        """int: Number of BBox in the current object."""
        return self.tensor.shape[0]

    def __repr__(self):
        """str: Return a strings that describes the object."""
        return self.__class__.__name__ + "(\n    " + str(self.tensor) + ")"