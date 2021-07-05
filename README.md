# Fre-GAN-pytorch (unofficial)

https://arxiv.org/abs/2106.02297

### Training

1. **한국어 음성 데이터 다운로드**

    * [KSS](https://www.kaggle.com/bryanpark/korean-single-speaker-speech-dataset)
    
2. pytorch_wavelets 설치

    * https://pytorch-wavelets.readthedocs.io/en/latest/readme.html#installation

3. **`~/Fre-GAN-pytorch`에 학습 데이터 준비**

   ```
   Fre-GAN-pytorch
     |- archive
         |- kss
             |- 1
             |- 2
             |- 3
             |- 4
         |- transcript.v.1.x.txt
   ```

4. **Preprocess**
   ```
   python preprocess.py
   ```
     * data 폴더에 학습에 필요한 파일들이 생성됩니다

5. **Train**
   ```
   python train.py -n name
   ```
   name을 원하는 이름으로 바꿔줍니다

   재학습 시
   ```
   python train.py -n name -p ./ckpt/name/ckpt-340000.pt
   ```
     * 불러올 모델 경로를 지정합니다

6. **Inference**
   ```
   python inference.py -p ./ckpt/name/ckpt-340000.pt
   ```
     * test 폴더에 wav 파일을 넣으면, 멜스펙트로그램으로 바꾼 후 fre-gan에 입력하고 output 폴더에 출력 wav가 생성됩니다



Reference
  * https://github.com/jik876/hifi-gan
