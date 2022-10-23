# [CVinW | ECCV 2022] How well does CLIP understand texture?

We investigate how well does CLIP understand texture in natural images described by natural language. To this end we analyze CLIP's ability to: (1) perform zero-shot learning on various texture and material classification datasets; (2) represent compositional properties of texture such as red dots or yellow stripes on the Describable Texture in Detail Dataset (DTD2); and (3) aid fine-grained categorization of birds in photographs described by color and texture of their body parts. 

## Dependency

1. **CLIP**: follow instructions in  https://github.com/openai/CLIP to download and install the package.
2. **DTD2**: download https://github.com/ChenyunWu/DescribingTextures as well as its data and trained model into `/your/path/to/dtd2`.
   Install the package locally by running `pip install /your/path/to/dtd2`.
3. **CUB Dataset**: can be downloaded from `http://www.vision.caltech.edu/visipedia/CUB-200-2011.html`
4. **FMD**: https://github.com/yinhaoz/denoising-fluorescence
5. **KTH-TIPS**/**KTH-TIPS2a**: https://www.csc.kth.se/cvap/databases/kth-tips/index.html

## Zero-Shot Classification Results

The table below shows the per-image accuracy using various image
encoders and prompts. Larger transformer-based models are better
(e.g., 83.4% vs. 90.5% accuracy on FMD for ViT-B/32 vs ViT-L/14 in Table
1), and prompt engineering makes a smaller difference for bigger
models (Table 2). CLIP achives remarkably good zero-shot performace on
all the four datasets.

**Table 1.** Accuracy of various image encoders for zero-shot
classificaton.

| Prompt                   |    Model     | DTD  | FMD  | KTH-TIPS | KTH-TIPS2a |
| ------------------------ | :----------: | :--: | :--: | :------: | :--------: |
| a photo of a {c} pattern |     RN50     | 40.7 | 83.4 |   49.1   |    62.8    |
| a photo of a {c} pattern |    RN101     | 42.0 | 79.0 |   48.5   |    51.3    |
| a photo of a {c} pattern |   ViT-B/32   | 41.1 | 83.8 |   58.4   |    59.5    |
| a photo of a {c} pattern |   ViT-B/16   | 44.7 | 87.9 |   57.4   |    61.1    |
| a photo of a {c} pattern |   ViT-L/14   | 50.4 | 89.5 |   63.5   |    64.5    |
| a photo of a {c} pattern | ViT-L/14@336 | 50.7 | 90.5 |   63.9   |    66.0    |


**Table 2.** Accuracy of various prompts used to encode the classes
for two different image encoders. Prompt engineering has a smaller
impact (lower variance) for the ViT-L/14 model.

| Prompt                      |  Model   | DTD  | FMD  | KTH-TIPS | KTH-TIPS2a |
| --------------------------- | :------: | :--: | :--: | :------: | :--------: |
| {c}                         | ViT-B/32 | 41.1 | 80.0 |   48.6   |    46.7    |
| a photo of a {c}            | ViT-B/32 | 43.1 | 79.9 |   50.4   |    49.9    |
| a photo of a {c} background | ViT-B/32 | 40.4 | 83.4 |   54.2   |    45.3    |
| a photo of a {c} object     | ViT-B/32 | 42.3 | 83.2 |   56.3   |    59.7    |
| a photo of a {c} pattern    | ViT-B/32 | 41.1 | 83.8 |   58.4   |    59.5    |
| {c}                         | ViT-L/14 | 50.4 | 88.7 |   58.3   |    68.0    |
| a photo of a {c}            | ViT-L/14 | 52.3 | 89.0 |   61.0   |    69.4    |
| a photo of a {c} background | ViT-L/14 | 50.4 | 89.3 |   59.3   |    69.8    |
| a photo of a {c} object     | ViT-L/14 | 53.0 | 92.3 |   59.6   |    70.0    |
| a photo of a {c} pattern    | ViT-L/14 | 50.4 | 89.5 |   63.5   |    64.5    |

**Table 3.** Further results using the ViT-L/14@336px model.

| Prompt                   |    Model     | DTD  | FMD  | KTH-TIPS | KTH-TIPS2a |
| ------------------------ | :----------: | :--: | :--: | :------: | :--------: |
| a photo of a {c} object  | Vit-L/14@366 | 53.3 | 93.6 |   59.4   |    69.5    |
| a photo of a {c} pattern | ViT-L/14@336 | 50.7 | 90.5 |   63.9   |    66.0    |
