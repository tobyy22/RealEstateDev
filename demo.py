# file: control_mlsd.py
import torch, cv2, einops, numpy as np, random
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.mlsd import MLSDdetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

class ControlMLSD:
    def __init__(self, cfg_path='./models/cldm_v15.yaml', ckpt_path='./models/control_sd15_mlsd.pth', device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() and device=='cuda' else 'cpu')
        self.apply_mlsd = MLSDdetector()
        self.model = create_model(cfg_path).to(self.device)
        self.model.load_state_dict(load_state_dict(ckpt_path, location=self.device.type))
        self.sampler = DDIMSampler(self.model)

    @torch.inference_mode()
    def detect_lines(self, img_bgr: np.ndarray, detect_resolution=512, value_threshold=0.1, distance_threshold=0.1):
        img = HWC3(img_bgr)
        det = self.apply_mlsd(resize_image(img, detect_resolution), value_threshold, distance_threshold)
        return HWC3(det)  # HxWx3 uint8

    @torch.inference_mode()
    def generate(
        self, input_image: np.ndarray, prompt: str,
        a_prompt='best quality, extremely detailed', n_prompt='lowres, bad anatomy, low quality',
        num_samples=1, image_resolution=512, ddim_steps=20, guess_mode=False, strength=1.0,
        scale=9.0, seed=-1, eta=0.0, value_threshold=0.1, distance_threshold=0.1
    ):
        img = HWC3(input_image)
        H0, W0, _ = resize_image(img, image_resolution).shape
        det = self.detect_lines(img, image_resolution, value_threshold, distance_threshold)
        det = cv2.resize(det, (W0, H0), interpolation=cv2.INTER_NEAREST)

        control = torch.from_numpy(det.copy()).float().to(self.device) / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').contiguous()

        if seed == -1:
            seed = random.randint(0, 2**15)
        seed_everything(seed)

        cond = {"c_concat": [control],
                "c_crossattn": [self.model.get_learned_conditioning([f"{prompt}, {a_prompt}"]*num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control],
                   "c_crossattn": [self.model.get_learned_conditioning([n_prompt]*num_samples)]}
        shape = (4, H0//8, W0//8)

        self.model.control_scales = (
            [strength * (0.825 ** float(12 - i)) for i in range(13)]
            if guess_mode else [strength]*13
        )

        samples, _ = self.sampler.sample(
            ddim_steps, num_samples, shape, cond, verbose=False, eta=eta,
            unconditional_guidance_scale=scale, unconditional_conditioning=un_cond
        )
        x = self.model.decode_first_stage(samples)
        x = (einops.rearrange(x, 'b c h w -> b h w c') * 127.5 + 127.5).clamp(0,255).cpu().numpy().astype(np.uint8)

        # return both control map and images
        ctrl_vis = 255 - cv2.dilate(det, np.ones((3,3), np.uint8), 1)
        return ctrl_vis, [x[i] for i in range(num_samples)]


pipe = ControlMLSD(device='cuda')  # or 'cpu'
img = cv2.imread("modern_room.jpg")
ctrl, outs = pipe.generate(img, prompt="modern nature room, raw wood, fireplace")
cv2.imwrite("control.png", ctrl)
for i, im in enumerate(outs): cv2.imwrite(f"out_{i}.png", im)