from tidecv import TIDE, datasets

tide = TIDE()
#tide.evaluate(datasets.COCO('/data0/kjj/dataset/mscoco/annotations/instances_val2017.json'), datasets.COCOResult('/data0/hb/mmdetection-2.18.0/val_dev_results/RetinaNet.json'), mode=TIDE.BOX)
tide.evaluate(datasets.COCO('/data0/kjj/dataset/mscoco/annotations/instances_val2017.json'), datasets.COCOResult('/data0/hb/mmdetection-2.18.0/val_dev/semi-decoupled_head.json'), mode=TIDE.BOX)
#tide.evaluate(datasets.COCO('/data0/kjj/dataset/mscoco/annotations/instances_val2017.json'), datasets.COCOResult('/data0/hb/mmdetection-2.18.0/demo/val_jsonfile/decoupled_head.json'), mode=TIDE.BOX)
tide.summarize()  # Summarize the results as tables in the console
tide.plot(out_dir='/data0/hb/mmdetection-2.18.0/demo/val_dev/')       # Show a summary figure. Specify a folder and it'll output a png to that folder.