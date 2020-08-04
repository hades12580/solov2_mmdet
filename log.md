

- Results:

  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.388
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.666
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.378
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.046
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.346
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.737
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.276
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.428
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.442
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.092
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.447
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.781
  +------------+--------+----------+--------+----------+--------+
  | category   | AP     | category | AP     | category | AP     |
  +------------+--------+----------+--------+----------+--------+
  | person     | 25.679 | rider    | 28.718 | car      | 47.424 |
  | truck      | 47.697 | bus      | 52.807 | train    | 49.958 |
  | motorcycle | 33.932 | bicycle  | 23.853 | None     | None   |
  +------------+--------+----------+--------+----------+--------+
  ```

  

- Cityscapes is overfited:

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.390
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.666
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.383
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.045
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.350
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.737
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.279
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.429
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.443
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.091
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.446
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.784
+------------+--------+----------+--------+----------+--------+
| category   | AP     | category | AP     | category | AP     |
+------------+--------+----------+--------+----------+--------+
| person     | 25.767 | rider    | 29.077 | car      | 47.442 |
| truck      | 47.484 | bus      | 52.339 | train    | 50.912 |
| motorcycle | 34.621 | bicycle  | 24.071 | None     | None   |
+------------+--------+----------+--------+----------+--------+

```

- Speed comparasion with num_grids double
  Before it, we get 6-8 fps;
  After: 5-6 fps