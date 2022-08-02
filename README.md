## Classification models by torchvision

### Project Structure
```
SimilarityModule
        |
        ├── logs
        |     ├──  LOG/
        |     └──  model/  
        |
        ├── utils
        |     ├──  choose_class_weight.py
        |     ├──  data_generator.py
        |     ├──  imbalanced.py
        |     ├──  loadjsonconfig.py
        |     ├──  models.py
        |     ├──  prepare_dataset.py
        |     ├──  simple_tools.py
        |     ├──  config.json
        |     └──  requirements.txt
        |
        ├── dataset
        |     └──  train
        |     |      ├──  class_1/
        |     |      ├──  class_2/
        |     |      ├──  class_3/
        |     |      ├──  ...
        |     |      └──  class_n/
        |     |
        |     ├──  val
        |     |      ├──  class_1/
        |     |      ├──  class_2/
        |     |      ├──  class_3/
        |     |      ├──  ...
        |     |      └──  class_n/
        |     |
        |     └──  test
        |            ├──  class_1/
        |            ├──  class_2/
        |            ├──  class_3/
        |            ├──  ...
        |            └──  class_n/
        |
        ├── eval.py
        ├── train.py
        └── README.md
```

### Usage
* **Train**

```bash
python train.py --jsonconfig_path "path to config file" --model_name="name of backbone"
```
Example:
```bash
python train.py --jsonconfig_path "config.json" --model_name="mobilenet_v2"
```

* **Evaluate**

```bash
python eval.py --jsonconfig_path "path to config file" --model_name="name of backbone" --model_path "path to model.pth"
```
Example:
```bash
python eval.py --jsonconfig_path "config.json" --model_name="mobilenet_v2" --model_path "E:\TaiLam_Bosco\a\logs\model\13.pth"
```