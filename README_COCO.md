Download and extract CoCo dataset from http://cocodataset.org/#download and install the python api. I'm using the MSCOCO2014. 

Checkout COCO_Splice branch

run python maskgen/batch/batch_project.py --COCO_flag --COCO_Dir /dvmm-filer2/users/xuzhang/Medifor/data/MSCOCO/ --count 10 --results ./synthesized_journals/ --json tests/batch_process_coco.json --loglevel 0 --auto_graph

--COCO_Dir is the directory to MSCOCO dataset. 
--count is the number of journals that will be generated
--results is the directory to store the result

The operations that are used to generate the graph are defined in ./tests/operation_pool.json