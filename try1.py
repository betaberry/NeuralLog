from neurallog import data_loader

log_file = "logs/BGL.log"


if __name__ == '__main__':
    (x_tr, y_tr), (x_te, y_te) = data_loader.load_supercomputers(
        log_file, train_ratio=0.8, windows_size=20,
        step_size=5, e_type='bert', mode='balance')
    
    print("-----ok-----")
    print(len(x_tr))
    print(len(y_tr))