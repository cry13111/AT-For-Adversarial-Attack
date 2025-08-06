This is AT-For-Adversarial-Attack

## 📋 Requirements
<pre>
  python version: 3.9.21 
  CUDA version: 12.1
</pre>

<pre>
  numpy==1.26.4
  torch==2.5.1
  torchvision==0.20.1
  timm==1.0.15
  tqdm==4.56.2
  opencv==4.10.0
  pillow==11.1.0
</pre>

## ⚙️ Preparation
<pre>
  conda create --name at_for_attack python==3.9
  conda activate at_for_attack
  pip install -r requirements.txt
</pre>

## ⬇️ Download Adversarial checkpoints
We use the adversarial training model provided as a pretrained model in [ARES 2.0](https://github.com/thu-ml/ares)

## 📂 File Explanation
- The `PGD_train` folder contains the code implementation of the PGD (Projected Gradient Descent) adversarial attack.

- The `attack.py` file is mainly responsible for generating adversarial examples.

- The `saliency_map.py` file is primarily used to obtain the saliency maps of the images.

- The `mask_perturbations.py` file implements the process of applying perturbations based on the saliency map masks.

## 🙏 Acknowledgments
For the implementation of the adversarial attack methods, we leveraged the [TransferAttack](https://github.com/Trustworthy-AI-Group/TransferAttack). We thank the authors for their contributions to the implementation of these attack methods.
