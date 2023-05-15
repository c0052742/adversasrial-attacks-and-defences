import torch
from art.defences.preprocessor import FeatureSqueezing

class YoloWithFeatureSqueezing(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.feature_squeezing = FeatureSqueezing(bit_depth=7, clip_values=(0, 1))

    def forward(self, x, targets=None):
        x_np, _ = self.feature_squeezing(x.cpu().numpy())  # Apply feature squeezing to the input images
        x = torch.from_numpy(x_np).to(self.model.device)
        
        if self.training:
            return self.model(x, targets)
        else:
            return self.model(x)

# Load the model
weights = "runs/train/KITTI_med_evolve/weights/best.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights, force_reload=True)
model.eval()

# Wrap the model with feature squeezing
model_with_feature_squeezing = YoloWithFeatureSqueezing(model)
model_with_feature_squeezing.to("cuda")
