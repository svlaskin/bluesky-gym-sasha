"""Variable-uncooperative wrappers for SectorCR uncooperative environments.

This module does not modify existing environment code. It provides new classes
that expose configurable ``n_uncoop`` values for ATT and non-ATT variants.
"""

import gymnasium as gym
import numpy as np

from bluesky_zoo.sector_cr import sector_cr_sas_uncooperative as base


class SectorCR_sas_uncoop_variable(base.SectorCR_sas_uncoop):
    """SectorCR uncooperative env with a safe configurable ``n_uncoop``."""

    def __init__(self, render_mode=None, n_agents=20, n_uncoop=3):
        n_agents = int(n_agents)
        requested_n_uncoop = max(0, min(int(n_uncoop), n_agents))

        # Base env uses self.agents[-n_uncoop:], where n_uncoop=0 would select all
        # agents because -0 == 0. Initialize with at least 1 then fix locally.
        init_n_uncoop = requested_n_uncoop if requested_n_uncoop > 0 else 1
        super().__init__(render_mode=render_mode, n_agents=n_agents, n_uncoop=init_n_uncoop)

        self.agents_uncoop = self.agents[-requested_n_uncoop:] if requested_n_uncoop else []
        self.n_uncoop = requested_n_uncoop


class SectorCR_ATT_sas_uncoop_variable(base.SectorCR_ATT_sas_uncoop):
    """ATT variant with configurable ``n_uncoop`` support."""

    def __init__(self, render_mode=None, n_agents=20, n_uncoop=2):
        n_agents = int(n_agents)
        requested_n_uncoop = max(0, min(int(n_uncoop), n_agents))

        # Call the base non-ATT constructor directly so we can pass n_uncoop.
        init_n_uncoop = requested_n_uncoop if requested_n_uncoop > 0 else 1
        base.SectorCR_sas_uncoop.__init__(
            self,
            render_mode=render_mode,
            n_agents=n_agents,
            n_uncoop=init_n_uncoop,
        )

        self.agents_uncoop = self.agents[-requested_n_uncoop:] if requested_n_uncoop else []
        self.n_uncoop = requested_n_uncoop

        # Keep the ATT observation-space shape used by the original ATT class.
        self.observation_spaces = {
            agent: gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64)
            for agent in self.agents
        }
