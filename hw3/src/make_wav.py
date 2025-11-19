#fluidsynth -F 300_0.wav -ni /usr/share/sounds/sf2/FluidR3_GM.sf2 output/300_0.mid
import os
import subprocess

# 根資料夾路徑（底下所有子資料夾都會被掃到）
root_folder = "output"   # 你放 MIDI 的總資料夾
sound_font = "/usr/share/sounds/sf2/FluidR3_GM.sf2"

for dirpath, dirnames, filenames in os.walk(root_folder):
    for filename in filenames:
        if filename.lower().endswith(".mid"):
            mid_path = os.path.join(dirpath, filename)

            # wav 檔名（同名不同副檔名）
            wav_name = filename.rsplit(".", 1)[0] + ".wav"
            wav_path = os.path.join(dirpath, wav_name)

            # fluidsynth 指令
            cmd = [
                "fluidsynth",
                "-F", wav_path,
                "-ni",
                sound_font,
                mid_path
            ]

            print("Running:", " ".join(cmd))
            subprocess.run(cmd, check=True)

print("全部轉換完成！")