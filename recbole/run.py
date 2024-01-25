from recbole.quick_start import run_recbole

# generate the interaction file first using gen_inter.py

run_recbole(model='BPR', config_file_list=['bpr-general.yml'])