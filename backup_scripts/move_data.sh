#!/bin/bash
python move_data.py --target_path /ist-nas/users/puntawatp/TPAMI_MajorRevision_data --json_file /home/mint/Dev/DiFaReli/difareli-faster/visualize_scripts/TPAMI/MajorRevision/user_study/rotateSH/user_study_rotateSH_axis2.json && 
python move_data.py --target_path /ist-nas/users/puntawatp/TPAMI_MajorRevision_data --json_file /home/mint/Dev/DiFaReli/difareli-faster/visualize_scripts/TPAMI/MajorRevision/user_study/rotateSH/user_study_rotateSH_axis1.json &&
python move_data.py --target_path /ist-nas/users/puntawatp/TPAMI_MajorRevision_data --json_file /home/mint/Dev/DiFaReli/difareli-faster/visualize_scripts/TPAMI/MajorRevision/user_study/targetSH/user_study_targetSH.json &&
python move_data.py --target_path /ist-nas/users/puntawatp/TPAMI_MajorRevision_data --json_file /home/mint/Dev/DiFaReli/difareli-faster/visualize_scripts/TPAMI/MajorRevision/figure/ffhq/targetSH.json &&
python move_data.py --target_path /ist-nas/users/puntawatp/TPAMI_MajorRevision_data --json_file /home/mint/Dev/DiFaReli/difareli-faster/visualize_scripts/TPAMI/MajorRevision/figure/ffhq/rotateSH_axis1.json &&
python move_data.py --target_path /ist-nas/users/puntawatp/TPAMI_MajorRevision_data --json_file /home/mint/Dev/DiFaReli/difareli-faster/visualize_scripts/TPAMI/MajorRevision/figure/ffhq/rotateSH_axis2.json

