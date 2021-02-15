
## TUM RGB-D SLAM Dataset and Benchmark

Scene [`long_office_household`](https://vision.in.tum.de/data/datasets/rgbd-dataset/download#freiburg3_long_office_household) is from Handheld SLAM category, while `pioneer_slam` and [`pioneer_slam3`](https://vision.in.tum.de/data/datasets/rgbd-dataset/download#freiburg2_pioneer_slam3) are from Robot SLAM category.

> The sequences have been recorded using either a Kinect1 or an Asus XTion, two popular RGBD sensors based on triangulation. Triangulation-based sensors are affected by noise with a different noise pattern than time-of-flight sensors

```
re_thre=2deg, te_thre=10cm, inlier_thre=15cm
****************************** FPFH + RANSAC ******************************
       SceneName         Valid Success Inlier% Inlier(#) Re(deg) Te(cm) Median  Mean  STD  FeatTime RegTime
      pioneer_slam       100%    47%    11.4%    78.4    0.44   47.14   0.08   0.08  0.03    0.08    0.06
     pioneer_slam3       100%    52%     5.3%   121.5    0.10   27.18   0.09   0.10  0.02    0.10    0.07
 long_office_household   100%    50%    13.4%    68.0    0.37   50.47   0.08   0.08  0.02    0.08    0.05
        Average          100%    50%    10.0%    89.3    0.30   41.60   0.09   0.09  0.03    0.09    0.06
****************************** FCGF(3DMatch) + RANSAC ******************************
       SceneName         Valid Success Inlier% Inlier(#) Re(deg) Te(cm) Median  Mean  STD  FeatTime RegTime
      pioneer_slam       100%    55%    18.3%   179.1    0.32   34.40   0.08   0.08  0.01    0.08    0.04
     pioneer_slam3       100%    66%    55.4%  1251.6    0.03    9.33   0.07   0.07  0.01    0.07    0.06
 long_office_household   100%    38%     8.7%    60.3    0.44   52.35   0.07   0.08  0.03    0.08    0.02
        Average          100%    53%    27.5%   497.0    0.26   32.02   0.07   0.08  0.02    0.08    0.04
****************************** FCGF(KITTI) + RANSAC ******************************
       SceneName         Valid Success Inlier% Inlier(#) Re(deg) Te(cm) Median  Mean  STD  FeatTime RegTime
      pioneer_slam       100%     1%    17.5%   152.2    1.97  471.36   0.07   0.07  0.01    0.07    0.07
     pioneer_slam3       100%     0%    52.4%  1185.8    1.99  679.70   0.07   0.07  0.01    0.07    0.10
 long_office_household   100%     0%     8.2%    54.5    1.99  325.98   0.07   0.08  0.03    0.08    0.04
        Average          100%     0%    26.0%   464.2    1.98  492.35   0.07   0.07  0.02    0.07    0.07
```

## ETH
> It contains two indoor scenes (apartment and stairs), five outdoor scenes (gazebo summer, gazebo winter, plain, wood summer and wood autumn) and a mixed one (hauptgebaude). It includes both structured and unstructured environments, and the indoor ones are not entirely static (there are walking people or furni- ture moved between scans).
```
re_thre=2deg, te_thre=10cm, inlier_thre=15cm
****************************** FPFH + RANSAC ******************************
       SceneName         Valid Success Inlier% Inlier(#) Re(deg) Te(cm) Median  Mean  STD  FeatTime RegTime
         plain           100%     7%     0.8%    85.1    0.50  398.50   0.16   0.16  0.01    0.16    0.16
      wood_summer        100%     0%     0.4%   140.3    1.34  765.07   0.45   0.45  0.04    0.45    1.88
      wood_autumn        100%     1%     0.3%   119.9    1.52  634.20   0.43   0.43  0.03    0.43    1.78
     gazebo_summer       100%    16%     0.7%   170.6    0.95  319.75   0.34   0.32  0.06    0.32    0.68
     gazebo_winter       100%    14%     0.7%   193.7    0.75  258.81   0.31   0.31  0.03    0.31    0.78
       apartment         100%    32%     2.4%   197.4    0.42  132.77   0.17   0.17  0.02    0.17    0.12
         stairs          100%    29%     1.6%   168.2    0.74  228.35   0.16   0.18  0.05    0.18    0.14
      hauptgebaude       100%    11%     1.0%   306.0    0.10  246.65   0.33   0.33  0.02    0.33    0.34
        Average          100%    14%     1.0%   172.7    0.79  373.01   0.31   0.29  0.11    0.29    0.73
****************************** FCGF(3DMatch) + RANSAC ******************************
       SceneName         Valid Success Inlier% Inlier(#) Re(deg) Te(cm) Median  Mean  STD  FeatTime RegTime
         plain           100%    29%     0.0%     4.0    0.12  157.34   1.83   1.93  0.60    1.93    0.26
      wood_summer        100%    44%     0.2%    60.8    0.09   41.25   2.52   2.57  0.30    2.57    1.76
      wood_autumn        100%    51%     0.3%    87.0    0.11   60.29   2.43   2.46  0.16    2.46    1.60
     gazebo_summer       100%    66%     0.1%    31.2    0.11   43.43   2.07   2.18  0.59    2.18    1.07
     gazebo_winter       100%    83%     0.3%    67.2    0.05   17.52   1.85   1.91  0.27    1.91    1.05
       apartment         100%    71%     0.4%    33.5    0.06   24.76   1.70   1.89  0.66    1.89    0.19
         stairs          100%    66%     0.2%    17.9    0.21   58.18   1.65   1.78  0.46    1.78    0.30
      hauptgebaude       100%    17%     0.1%    28.5    0.03  200.43   2.11   2.13  0.16    2.13    1.34
        Average          100%    53%     0.2%    41.3    0.10   75.40   2.08   2.11  0.52    2.11    0.95
****************************** FCGF(KITTI) + RANSAC ******************************
```

## KAIST Lidar Autonomous Driving Dataset.

> The point clouds we are going to use for our bench- mark have been acquired with two Velodyne VLP-16 LiDARs mounted on the left and right of the top of a car.

```
re_thre=2deg, te_thre=10cm, inlier_thre=15cm
****************************** FPFH + RANSAC ******************************
       SceneName         Valid Success Inlier% Inlier(#) Re(deg) Te(cm) Median  Mean  STD  FeatTime RegTime
        urban05          100%    53%     0.0%     2.9    0.43 451520800.76   0.06   0.06  0.02    0.06    0.15
        Average          100%    53%     0.0%     2.9    0.43 451520800.76   0.06   0.06  0.02    0.06    0.15
****************************** FCGF(3DMatch) + RANSAC ******************************
       SceneName         Valid Success Inlier% Inlier(#) Re(deg) Te(cm) Median  Mean  STD  FeatTime RegTime
        urban05          100%    34%     0.9%    62.1    1.04 709834668.23   0.06   0.06  0.03    0.06    0.15
        Average          100%    34%     0.9%    62.1    1.04 709834668.23   0.06   0.06  0.03    0.06    0.15
****************************** FCGF(KITTI) + RANSAC ******************************
       SceneName         Valid Success Inlier% Inlier(#) Re(deg) Te(cm) Median  Mean  STD  FeatTime RegTime
        urban05          100%    49%     0.8%    60.0    0.58 566155901.68   0.06   0.06  0.03    0.06    0.19
        Average          100%    49%     0.8%    60.0    0.58 566155901.68   0.06   0.06  0.03    0.06    0.19
```