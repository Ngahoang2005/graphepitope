#preprocess.py
import os
import numpy as np
from transformers import AutoTokenizer, EsmModel
import torch
# Khởi tạo SaProt (Bạn nên để global hoặc truyền vào hàm để tránh load nhiều lần)
saprot_tokenizer = AutoTokenizer.from_pretrained("westlake-repl/SaProt_650M_AF2")
saprot_model = EsmModel.from_pretrained("westlake-repl/SaProt_650M_AF2")
DICT={
    'ALA': 'A', 'CYS': 'C', 'CCS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'MSE': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
}
def pdb_split(line):
    order=int(line[6:11].strip())
    atom=line[11:16].strip()
    amino=line[16:21].strip()
    chain=line[21]
    site=line[22:28].strip()
    x=line[28:38].strip()
    y=line[38:46].strip()
    z=line[46:54].strip()
    return order,atom,amino,chain,site,x,y,z
def judge(line,filt_atom='CA'):
    kind=line[:6].strip()
    if kind not in ['HETATM','ATOM']:
        return None
    order,atom,amino,chain,site,x,y,z=pdb_split(line)
    if filt_atom is not None and atom!=filt_atom:
        return None
    prefix=''
    if len(amino)>3:
        prefix=amino[0]
        amino=amino[-3:]
    if amino=='MSE':
        amino='MET'
    elif amino=='CCS' or amino[:-1]=='CS':
        amino='CYS'
    elif amino not in DICT.keys():
        return None
    return prefix+amino,chain,site,float(x),float(y),float(z)
# def process_dssp(dssp_file):
#     aa_type = "ACDEFGHIKLMNPQRSTVWY"
#     SS_type = "HBEGITSC"
#     rASA_std = [115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
#                 185, 160, 145, 180, 225, 115, 140, 155, 255, 230]
#     with open(dssp_file, "r") as f:
#         lines = f.readlines()
#     seq = ""
#     dssp_feature = []
#     position = []
#     p = 0
#     while lines[p].strip()[0] != "#":
#         p += 1
#     for i in range(p + 1, len(lines)):
#         aa = lines[i][13]
#         if aa == "!" or aa == "*":
#             continue
#         seq += aa
#         POS = lines[i][5:11].strip()
#         position.append(POS)
#         SS = lines[i][16]
#         if SS == " ":
#             SS = "C"
#         SS_vec = np.zeros(8)
#         SS_vec[SS_type.find(SS)] = 1
#         PHI = float(lines[i][103:109].strip())
#         PSI = float(lines[i][109:115].strip())
#         ACC = float(lines[i][34:38].strip())
#         ASA = min(100, round(ACC / rASA_std[aa_type.find(aa)] * 100)) / 100
#         dssp_feature.append(np.concatenate((np.array([PHI, PSI, ASA]), SS_vec)))

#     return seq, dssp_feature,position
# def transform_dssp(dssp_feature):
#     dssp_feature = np.array(dssp_feature)
#     angle = dssp_feature[:,0:2]
#     ASA_SS = dssp_feature[:,2:]
#     radian = angle * (np.pi / 180)
#     dssp_feature = np.concatenate([np.sin(radian), np.cos(radian), ASA_SS], axis = 1)
#     return dssp_feature
# def get_dssp(ID,root):
#     if not os.path.exists(f"{root}/dssp/"):
#         os.mkdir(f"{root}/dssp/")
#     os.system(f"./mkdssp/mkdssp -i {root}/purePDB/{ID}.pdb -o {root}/dssp/{ID}.dssp")
#     if not os.path.exists(f"{root}/dssp/" + ID + ".dssp"):
#         return None
#     dssp_seq, dssp_matrix,position = process_dssp(f"{root}/dssp/" + ID + ".dssp")
#     np.save(f"{root}/dssp/" + ID, transform_dssp(dssp_matrix))
#     np.save(f"{root}/dssp/"+ID+"_pos",position)

def get_foldseek_3di(pdb_path):
    import subprocess
    import os
    import shutil

    if not os.path.exists(pdb_path) or os.path.getsize(pdb_path) == 0:
        return None

    tmp_dir = pdb_path + "_tmpbin"
    if os.path.exists(tmp_dir): shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)
    
    out_db = os.path.join(tmp_dir, "output")
    out_fasta = os.path.join(tmp_dir, "output.fasta")
    
    try:
        # Bước 1: Tạo DB từ file PDB
        subprocess.run(f"foldseek createdb {pdb_path} {out_db} -v 0", shell=True, check=True)
        
        # Bước 2: Dùng convert2fasta để xuất chuỗi 3Di ra file FASTA
        # Tham số --db-output 1 sẽ lấy chuỗi 3Di thay vì Amino Acid
        subprocess.run(f"foldseek convert2fasta {out_db} {out_fasta} -v 0", shell=True, check=True)
        
        # Bước 3: Đọc file FASTA để lấy chuỗi
        if os.path.exists(out_fasta):
            with open(out_fasta, "r") as f:
                lines = f.readlines()
                # File FASTA: dòng 1 là >Header, dòng 2 là Sequence
                if len(lines) >= 2:
                    return lines[1].strip()
    except Exception as e:
        # print(f"Foldseek error: {e}")
        return None
    finally:
        if os.path.exists(tmp_dir): shutil.rmtree(tmp_dir)
            
    return None
def extract_saprot_feat(pdb_id, amino_seq, root, device='cuda'):
    pdb_path = f"{root}/purePDB/{pdb_id}.pdb"
    seq_3di = get_foldseek_3di(pdb_path)
    if seq_3di is None: return None
    
    # Format SaProt: kết hợp axit amin (chữ thường) và 3Di (chữ hoa)
    # Ví dụ: "aV lA sS..."
    min_len = min(len(amino_seq), len(seq_3di))
    a_seq = amino_seq[:min_len]
    s_seq = seq_3di[:min_len]
    combined_seq = " ".join([f"{a.lower()}{s.upper()}" for a, s in zip(a_seq, s_seq)])
    
    saprot_model.to(device)
    inputs = saprot_tokenizer(combined_seq, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = saprot_model(**inputs)
    
    # Lấy embedding, bỏ BOS/EOS tokens ([1:-1])
    return outputs.last_hidden_state[0, 1:-1, :].cpu()

def get_saprot(name, amino_seq, root, device='cuda'):
    """
    Hàm điều phối: Kiểm tra file tồn tại, trích xuất và lưu SaProt embedding
    Tương đương với get_dssp cũ.
    """
    # 1. Tạo thư mục saprot nếu chưa có
    save_path = f"{root}/saprot"
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    target_file = f"{save_path}/{name}.pt"
    
    # 2. Nếu file đã tồn tại thì bỏ qua (tiết kiệm thời gian)
    if os.path.exists(target_file):
        return
    
    # 3. Trích xuất đặc trưng
    # Chú ý: pdb_id ở đây thường là tên file PDB trong purePDB (ví dụ: 1A2Z_A)
    feat = extract_saprot_feat(name, amino_seq, root, device)
    
    if feat is not None:
        # 4. Lưu file. Dùng torch.save vì feat là Tensor.
        # Nếu muốn dùng np.save như DSSP cũ thì dùng: np.save(target_file, feat.numpy())
        torch.save(feat, target_file)
    else:
        print(f"⚠️ Cảnh báo: Không thể trích xuất SaProt cho {name}")