"""Validated parameters for the interactive nuclear transport module."""
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict

class NuclearParams(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    model:Literal['transport']='transport'
    transport_duration:float=Field(260,ge=30,le=600)
    droplet_count:int=Field(28,ge=8,le=64)
    capture_strength:float=Field(2,ge=.1,le=4)
    transport_velocity:float=Field(1.2,ge=0,le=3)
    membrane_permeability:float=Field(1.4,ge=.05,le=4)
    nuclear_permeability:float=Field(1.2,ge=.05,le=4)
    growth_rate:float=Field(.14,ge=0,le=.4)
    brownian:float=Field(.9,ge=0,le=3)
    gate_spacing:int=Field(40,ge=24,le=64)
    entry_half_width:float=Field(4,ge=2,le=10)
    seed:int=Field(7,ge=0,le=2147483647)

def simulate_nuclear(p,out,progress):
    from droplet_transport import simulate_transport
    return simulate_transport(p,out,progress)
