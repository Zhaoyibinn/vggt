# Colmap Demo

This is a reconstruction demo for Colmap format with camera pose and intrinsic.
```
python main.py
```
The environment is as same as shown in README_origin.md.

## Dataset

The current dataset adopts the colmap format DTU dataset and contains two scenes, each with three images, as shown in `demo_comap`.


After testing, VGGT performs well on `scan24`, but there is an offset on `scan37`.


The final visualization results are stored in `test_with_gt.ply`, with green points representing Colmap results and red points representing VGGT results.

## Modification

The main modification is focused on the `run_demo` function of `demo_colmap.py`. When detecting the input camera `gt_extrinsic` and `gt_intrinsic`, it will read the colmap format pose and perform registration by roma.