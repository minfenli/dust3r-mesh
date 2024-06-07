# DUSt3R-Mesh: Geometric 3D Mesh Made Easy

![demo](assets/dust3r-mesh.jpg)

An implementation that utilize `DUSt3R: Geometric 3D Vision Made Easy`  
[[Project page](https://dust3r.europe.naverlabs.com/)] to reconstruct 3D meshes for unknown-pose images.


## Get Started
Please follow [DUSt3R](https://github.com/naver/dust3r) to prepare your enviorment.

You may also need to install open3d for mesh processing.
```python
pip install open3d
```

## Demo
```python
python mesh.py
```
Or your customized data path
```python
python mesh.py --data_path {your data path} --save_path {your save path}
```


## Reference

```bibtex
@inproceedings{dust3r_cvpr24,
      title={DUSt3R: Geometric 3D Vision Made Easy}, 
      author={Shuzhe Wang and Vincent Leroy and Yohann Cabon and Boris Chidlovskii and Jerome Revaud},
      booktitle = {CVPR},
      year = {2024}
}

@misc{dust3r_arxiv23,
      title={DUSt3R: Geometric 3D Vision Made Easy}, 
      author={Shuzhe Wang and Vincent Leroy and Yohann Cabon and Boris Chidlovskii and Jerome Revaud},
      year={2023},
      eprint={2312.14132},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## License

The code is distributed under the CC BY-NC-SA 4.0 License.
See [LICENSE](LICENSE) for more information.