"""Pair-only evidence interfaces behind an immutable 0/1/2 router."""
from pathlib import Path
import json
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SHIPSEAR = Path(r'D:\LSJ\Code\UART\shipsear')
VERIFY = SHIPSEAR/'ShipsEar_ComponentSet_Verification_20261022'
COUNT_STUDY = SHIPSEAR/'ShipsEar_CountIntervention_20261022'
sys.path.insert(0,str(VERIFY))
from verification_model import FrozenBranches, FixedDataset, MEMBERS, COUNTS, REFERENCE
from data_interface import recognition_metrics, recording_results
sys.path.insert(0,str(COUNT_STUDY))
from count_model import QuantityExpert, INITIAL, DATA_ROOT, strict_route

FIXED_CACHE = VERIFY/'runs/component_set/cache'
ROUTE_CACHE = COUNT_STUDY/'runs/count_intervention_seed13'
ARMS = ('G1_mlp','G2_forward','G3_bidirectional')


class PairInputs(nn.Module):
    def __init__(self,width=64):
        super().__init__()
        self.register_buffer('members',torch.from_numpy(MEMBERS[4:7]))
        self.context = nn.Sequential(nn.LayerNorm(1088),nn.Linear(1088,128),nn.GELU(),
                                     nn.Dropout(.1),nn.Linear(128,width),nn.GELU())
        self.component = nn.Sequential(nn.LayerNorm(192+width),nn.Linear(192+width,width),nn.GELU())
        self.presence = nn.Linear(width,1)
        # Every arm receives identical class identities, pair memberships and raw expert outputs.
        self.hypothesis = nn.Sequential(nn.LayerNorm(width+3+3+7+1+3),
                                        nn.Linear(width+3+3+7+1+3,width),nn.GELU())
        self.identity = nn.Parameter(torch.randn(3,width)*.02)

    def forward(self,data):
        context = self.context(data['context'])
        u = self.component(torch.cat((data['queries'],context[:,None].expand(-1,3,-1)),-1))
        prior = F.log_softmax(data['expert'][:,4:7],-1)
        common = torch.cat((context,data['log_count'].clamp(-12,0),
                            data['log_single'].clamp(-12,0),data['expert'].clamp(-12,0)),-1)
        v = self.hypothesis(torch.cat((common[:,None].expand(-1,3,-1),prior[...,None],
                                       self.members[None].expand(len(u),-1,-1)),-1))+self.identity[None]
        return u,v,self.presence(u).squeeze(-1),prior


class PairHead(nn.Module):
    def __init__(self,arm='G2_forward',width=64):
        super().__init__()
        if arm not in (*ARMS,'G4_support_only'):
            raise ValueError(arm)
        self.config = dict(arm=arm,width=width)
        self.arm = arm
        self.inputs = PairInputs(width)
        if arm=='G1_mlp':
            # Similar total parameter budget to the explicit forward interaction.
            self.fusion = nn.Sequential(nn.LayerNorm(6*width+6),nn.Linear(6*width+6,44),
                                       nn.GELU(),nn.Dropout(.1),nn.Linear(44,3))
            nn.init.zeros_(self.fusion[-1].weight)
            nn.init.zeros_(self.fusion[-1].bias)
        else:
            # Construct forward path before optional reverse path for identical G2/G3 tensors.
            self.forward_support = nn.Linear(width,width)
            self.forward_refute = nn.Linear(width,width)
            self.forward_gate = nn.Linear(2*width+1,2)
            self.forward_update = nn.Sequential(nn.Linear(width,width),nn.LayerNorm(width))
            self.forward_strength = nn.Parameter(torch.tensor(-2.))
            self.score = nn.Linear(width,1)
            nn.init.zeros_(self.score.weight)
            nn.init.zeros_(self.score.bias)
            if arm=='G3_bidirectional':
                self.reverse_support = nn.Linear(width,width)
                self.reverse_refute = nn.Linear(width,width)
                self.reverse_gate = nn.Linear(2*width+1,2)
                self.reverse_update = nn.Sequential(nn.Linear(width,width),nn.LayerNorm(width))
                self.reverse_strength = nn.Parameter(torch.tensor(-2.))

    def forward(self,data,mask_messages=False):
        u,v,raw_presence,prior = self.inputs(data)
        if self.arm=='G1_mlp':
            flat = torch.cat((u.flatten(1),v.flatten(1),raw_presence,prior),-1)
            delta = self.fusion(flat)
            return dict(scores=prior+delta,raw_presence=raw_presence,presence=raw_presence,
                        prior=prior,delta=delta)
        m = self.inputs.members[None,:,:,None]
        relation = 2*m-1
        ve = v[:,:,None].expand(-1,-1,3,-1)
        ue = u[:,None].expand(-1,3,-1,-1)
        if self.arm=='G3_bidirectional' and not mask_messages:
            edge = torch.cat((ue,ve,relation.expand(len(u),-1,-1,-1)),-1)
            gate = self.reverse_gate(edge).sigmoid()
            message = (prior.softmax(-1)[:,:,None,None]*(m*gate[...,:1]*self.reverse_support(ve)
                        -(1-m)*gate[...,1:]*self.reverse_refute(ve))).sum(1)
            u = u+self.reverse_strength.sigmoid()*self.reverse_update(message)
        presence = self.inputs.presence(u).squeeze(-1)
        p = presence.sigmoid()[:,None,:,None]
        # Correct candidate membership: present members and absent nonmembers support a pair.
        support = m*p+(1-m)*(1-p)
        refute = 1-support
        ue = u[:,None].expand(-1,3,-1,-1)
        edge = torch.cat((ue,ve,relation.expand(len(u),-1,-1,-1)),-1)
        gate = self.forward_gate(edge).sigmoid()
        if self.arm=='G4_support_only':
            # All members use presence; absence and all opposing messages are removed.
            message = (m*p*gate[...,:1]*self.forward_support(ue)).mean(2)
        else:
            message = (support*gate[...,:1]*self.forward_support(ue)
                       -refute*gate[...,1:]*self.forward_refute(ue)).mean(2)
        updated_v = v if mask_messages else v+self.forward_strength.sigmoid()*self.forward_update(message)
        delta = self.score(updated_v).squeeze(-1)
        return dict(scores=prior+delta,raw_presence=raw_presence,presence=presence,prior=prior,delta=delta)


class PairRecognizer:
    def __init__(self,manifest,device='cuda'):
        path = Path(manifest).resolve()
        info = json.loads(path.read_text(encoding='utf-8'))
        self.device = device
        self.branches = FrozenBranches(info['branch_reference'],device)
        self.quantity = QuantityExpert().to(device)
        saved = torch.load(info['quantity_checkpoint'],map_location='cpu',weights_only=False)
        self.quantity.load_state_dict(saved['model'])
        self.quantity.requires_grad_(False).eval()
        saved = torch.load(path.parent/info['checkpoint'],map_location='cpu',weights_only=False)
        self.head = PairHead(**saved['config']).to(device)
        self.head.load_state_dict(saved['model'])
        self.head.requires_grad_(False).eval()

    @torch.inference_mode()
    def predict(self,waves):
        waves = np.asarray(waves,dtype=np.float32)
        if waves.ndim==1:
            waves = waves[None]
        if waves.ndim!=2 or waves.shape[1]!=80000:
            raise ValueError('Expected 16-kHz audio [batch,80000]')
        features = self.branches.encode(waves)
        output = self.head(features)
        spectra,demon = self.quantity.frontend(torch.as_tensor(waves,device=self.device))
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logq = self.quantity(spectra.half().float(),demon.half().float()).cpu().numpy()
        result = {k:v.cpu().numpy() for k,v in output.items()}
        result['log_count'] = logq
        result['predictions'] = strict_route(logq,features['expert'][:,1:4].cpu().numpy(),result['scores'])
        return result
