

*2020.06.31*:

Continue exp on ghostnet with 2 altinatives:
- extract layer from before the last one;
- try add SPP module to the last extracted backbone feature;


*2020.06.30*:

Exp on ghostnet backbone, the mAP got best 80.4, but the mainly issue is that:

- detection unstable, sometimes clear and close target will miss (this should not be happen);
- detected objects too small, lack of global context (a small box detected inside a target), we may want expand the receptive filed;


*2020.06.24*: 
I am testing new model with PANet support.

```
Class      Images     Targets           P           R      mAP@.5  mAP@.5:.95: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 225/225 [00:24<00:00,  9.01it/s]
    all     1.8e+03     6.6e+03        0.53       0.854       0.803       0.588
```
seems not as good as previous one.

v1 version only using 3 epoch get 0.804 mAP:

```
  Images     Targets           P           R      mAP@.5  mAP@.5:.95: 100%|2/299██████████████████████████████████████████████████████████████████████████████████████████████████████████| 225/225 [00:27<00:00,  8.24it/s]
                 all     1.8e+03     6.6e+03       0.533       0.822       0.804       0.506
```