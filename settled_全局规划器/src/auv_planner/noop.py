from __future__ import annotations

import numpy as np

from .contracts import (
    CapabilitySet,
    ControlProposal,
    LocalObservation,
    Reference,
    SafeCommand,
    SafetyContext,
)


class NoLocalPlanner:
    def act(
        self,
        observation: LocalObservation,
        reference: Reference,
        capability: CapabilitySet,
    ) -> ControlProposal:
        del reference, capability
        return ControlProposal(
            stamp_s=observation.stamp_s,
            linear_accel_body_mps2=np.zeros(3, dtype=np.float64),
            angular_velocity_body_rps=np.zeros(3, dtype=np.float64),
            valid_until_s=observation.stamp_s + 0.1,
        )


class NoSafetyFilter:
    def filter(self, proposal: ControlProposal, context: SafetyContext) -> SafeCommand:
        del context
        return SafeCommand(command=proposal, allowed=True)
