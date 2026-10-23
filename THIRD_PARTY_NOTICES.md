# Third-party notices and distribution scope

The original code and documentation in this repository are licensed under **GNU Affero General Public License version 3 only (`AGPL-3.0-only`)**, except where a file explicitly states otherwise. [LICENSE](LICENSE) contains the complete GNU text, retrieved from [GNU](https://www.gnu.org/licenses/agpl-3.0.txt). This does not relicense datasets, pretrained weights, upstream repositories or installed dependencies.

## Dependencies and integrations

| Component | How it is used | Upstream terms/source |
|---|---|---|
| Ultralytics 8.4.182 | Optional YOLO semantic/detection training and inference dependency; no vendored distribution | [License choices and obligations](https://www.ultralytics.com/license), [source](https://github.com/ultralytics/ultralytics) |
| PyTorch / TorchVision | Installed dependencies for tensor operations and model training | [PyTorch license](https://github.com/pytorch/pytorch/blob/main/LICENSE), [TorchVision license](https://github.com/pytorch/vision/blob/main/LICENSE) |
| NumPy / Pillow | Installed image/array dependencies | [NumPy license](https://github.com/numpy/numpy/blob/main/LICENSE.txt), [Pillow license](https://github.com/python-pillow/Pillow/blob/main/LICENSE) |
| OpenCV, requests, PyYAML, psutil, pyserial and optional plotting/NVML libraries | Installed optional/runtime/test dependencies | Their distributions retain their own copyright and license files; see the installed package metadata |
| BEVFormer | Wrapper for a separately acquired upstream checkout; upstream code/weights are not bundled | [Apache-2.0 upstream license](https://github.com/fundamentalvision/BEVFormer/blob/master/LICENSE); recorded reference commit `66b65f3a1f58caf0507cb2a971b9c0e7f842376c` |
| GNU AGPL license text | Verbatim license document | Copyright 2007 Free Software Foundation, Inc.; verbatim copying permitted in its header |

Ultralytics offers AGPL and Enterprise options. This public source distribution uses the AGPL route; it is not evidence that a separate closed-source product has an Enterprise license. AGPL is not a blanket prohibition on commercial use. Distribution, modification and network-use obligations must be considered for the actual use. The upstream projects do not endorse this repository or its performance claims.

## Data and model weights

**No training/validation dataset image, annotation, derived mask, dataset archive, pretrained weight or locally trained checkpoint is included.** The simulator map extract described below is a separately licensed, explicitly bundled exception to the data-exclusion scope. Raw per-image manifests, contracts and experiment logs are also omitted because they contain source-derived information and local paths. Public evidence is limited to aggregate metrics, protocol details and content hashes.

The experiments used AI Hub, BDD100K, Virtual KITTI 2, CamVid, KITTI Road and public RoadObstacle21/RoadAnomaly21 subsets. See [DATASETS](docs/DATASETS.md) for provider links and restrictions. Adapters support additional sources, but support does not mean that data was acquired, trained on or cleared for redistribution.

AI Hub's FAQ distinguishes generated AI outputs from original data and simple processed copies. This repository excludes the original and processed data. Virtual KITTI 2 is offered for non-commercial use under CC BY-NC-SA 3.0. The combination of code licensing and dataset terms must be evaluated separately; the project's AGPL code license grants no rights to the underlying images or to the unpublished mixed-data checkpoints.

Download data and official pretrained assets directly from an authorized source, retain the applicable terms and attribution, and record the version/hash. Do not infer permission from a mirror, a public URL, a successful download, a published hash or this repository's license. Additional redistribution of model weights requires its own provenance and rights review.

## OpenStreetMap simulator region

The Seoul Seongsu map files in `src/camera_rc/simulator/assets/maps/` contain OpenStreetMap-derived road geometry and building footprints, © OpenStreetMap contributors. The data and adapted geographic database are distributed under the **Open Database License 1.0 (ODbL-1.0)**, not the code's AGPL license. See the [OSM copyright and attribution page](https://www.openstreetmap.org/copyright) and the [full ODbL license](https://opendatacommons.org/licenses/odbl/1-0/).

The bundled source-geometry extract, download query, provider snapshot time, response hash and projection/assumption rules allow inspection of the geographic derivation. Raw API responses are retained locally; author/contact/address metadata are not needed or bundled. Public geographic coordinates identify the selected simulation region, not the user's location. No external map tile service is used at runtime.

Road widths and building heights without source measurements are simulator assumptions. Spawned vehicles, pedestrians, cones, timing and their motion are synthetic test fixtures, not observed Seoul traffic. The source data and its contributors do not endorse this simulator or its safety claims.
