# Scyclone
A part of pytorch implementation of the [Scyclone](https://arxiv.org/abs/2005.03334)

Now buliding!

**Attention please!**

**This project just for practice, maybe there are a lot of bugs ~**

**Something update:**
- Add hydra-core config
- Add distributed training

**Something different from origin paper:**

- Use 80-dim mel spectrum instead of linear spectrum.
- Use AdaBeilef optimizer instead of Adam.

**For vocoder:**
- WaveRNN
- HIFIGAN
- MelGAN


**requirements:**
- torch >= 1.1.0
- hydra-core >= 1.0
- librosa