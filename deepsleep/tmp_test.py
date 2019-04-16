from deepsleep.data_loader import NonSeqMESADataLoader as DataLoader
DATA_PATH = "E:\\NCLOneDrive\\OneDrive - Newcastle University\\Dataset\\MESA\Acc-" \
            "HR-Aligned\\Aligned_final\\combined\\Aligned_final";
# data_file = "C:\\Users\\BB\\AnacondaProjects\\JJPPL\\example_data.csv"
SAVE_PATH = "C:\\tmp\\temp_result"
MAT_PATH = "D:\\Google\\AllCode\\issmp\\WorkInProgress\\Matlab\\MESA_ACC_PSG_Dataset.mat"
NPZ_PATH = 'D:\\tmp\\combined_seq2label0-300.npz'
data_loader = DataLoader(NPZ_PATH, SAVE_PATH)