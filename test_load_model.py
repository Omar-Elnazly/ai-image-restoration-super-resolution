import torch, sys, numpy as np
sys.path.append('.')
from inference.inference_pipeline import InferencePipeline
from PIL import Image

checkpoints = {
    'srcnn':     'checkpoints/srcnn/srcnn_best.pth',
    'denoising': 'checkpoints/denoising/denoising_best.pth',
    'srresnet':  'checkpoints/srresnet/srresnet_best.pth',
    'srgan':     'checkpoints/srgan/srgan_gen_epoch_0090.pth',
}

test_img = Image.fromarray(np.random.randint(0,255,(64,64,3),dtype=np.uint8))

print()
for name, ckpt in checkpoints.items():
    try:
        p = InferencePipeline(name, ckpt)
        result = p.run(test_img)
        out_size = result['output'].size
        print(f'  [OK]   {name:<12} output size: {out_size}')
    except Exception as e:
        print(f'  [FAIL] {name:<12} {e}')
print()