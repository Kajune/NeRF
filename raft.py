import torch
from torchvision.models.optical_flow import raft_large, raft_small, Raft_Large_Weights, Raft_Small_Weights


class RAFT:
	def __init__(self, model="small"):
		if model == "small":
			self.model = raft_small(Raft_Small_Weights.DEFAULT).cuda()
			self.transforms = Raft_Small_Weights.DEFAULT.transforms()
		else:
			self.model = raft_large(Raft_Large_Weights.DEFAULT).cuda()
			self.transforms = Raft_Large_Weights.DEFAULT.transforms()
		self.model.eval()


	def predict(self, imgL, imgR):
		with torch.no_grad():
			imgL = torch.FloatTensor(imgL.transpose(2,0,1)).unsqueeze(0)
			imgR = torch.FloatTensor(imgR.transpose(2,0,1)).unsqueeze(0)
			imgL, imgR = self.transforms(imgL, imgR)
			imgL = imgL.cuda()
			imgR = imgR.cuda()
			flow = self.model(imgL, imgR)[-1]
			return flow.cpu().numpy()[0]
