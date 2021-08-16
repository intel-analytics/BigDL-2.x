from pytorch_lightning.plugins.environments import ClusterEnvironment
from pytorch_lightning.utilities import rank_zero_only


class RayEnvironment(ClusterEnvironment):
    """Environment for PTL training on a Ray cluster."""

    def __init__(self, world_size):
        self.set_world_size(world_size)
        self._global_rank = 0
        self._is_remote = False

    def creates_children(self) -> bool:
        return False

    def master_address(self) -> str:
        raise NotImplementedError

    def master_port(self) -> int:
        raise NotImplementedError

    def world_size(self) -> int:
        return self._world_size

    def set_world_size(self, size: int) -> None:
        self._world_size = size

    def global_rank(self) -> int:
        return self._global_rank

    def set_global_rank(self, rank: int) -> None:
        self._global_rank = rank
        rank_zero_only.rank = rank  # type: ignore

    def set_remote_execution(self, is_remote: bool) -> None:
        self._is_remote = is_remote

    def is_remote(self) -> bool:
        return self._is_remote

    def local_rank(self) -> int:
        raise NotImplementedError

    def node_rank(self) -> int:
        raise NotImplementedError
