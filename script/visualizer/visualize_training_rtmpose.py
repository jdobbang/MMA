import pandas as pd
import matplotlib.pyplot as plt
import json
import os

def load_and_plot_logs(file_path, save_name="training_report.png"):
    """
    json 파일을 읽어서 학습 그래프를 생성하고 저장합니다.
    """
    # 1. 파일에서 데이터 읽기
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            # 방식 A: 파일 전체가 하나의 JSON 리스트인 경우
            data = json.load(f)
        except json.JSONDecodeError:
            # 방식 B: 한 줄당 하나의 JSON 객체가 있는 경우 (JSONL)
            f.seek(0)
            data = [json.loads(line) for line in f if line.strip()]

    # 2. 데이터프레임 변환 및 정렬
    df = pd.DataFrame(data)
    df = df.sort_values('step')

    # 3. 그래프 그리기 설정
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plt.subplots_adjust(top=0.90, hspace=0.3)

    last_step = df['step'].iloc[-1]
    last_loss = df['loss'].iloc[-1] if 'loss' in df.columns else None
    last_acc = df['acc_pose'].iloc[-1] if 'acc_pose' in df.columns else None
    last_ap = df['coco/AP'].iloc[-1] if 'coco/AP' in df.columns else None

    title_str = f'Training Report (Final Step: {last_step})'
    if last_loss is not None:
        title_str += f'\nLoss: {last_loss:.4f}'
    if last_acc is not None:
        title_str += f' | Acc: {last_acc:.4f}'
    if last_ap is not None:
        title_str += f' | AP: {last_ap:.4f}'

    fig.suptitle(title_str, fontsize=16)

    # 그래프 1: Loss 변화량
    if 'loss' in df.columns:
        axes[0, 0].plot(df['step'], df['loss'], color='tab:red', label='Loss', linewidth=2)
        axes[0, 0].set_title('Training Loss Trend')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
    else:
        axes[0, 0].text(0.5, 0.5, 'No Loss Data', ha='center', va='center', transform=axes[0, 0].transAxes)

    # 그래프 2: Pose Accuracy 변화량
    if 'acc_pose' in df.columns:
        axes[0, 1].plot(df['step'], df['acc_pose'], color='tab:blue', label='Accuracy', linewidth=2)
        axes[0, 1].set_title('Pose Accuracy Trend')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
    else:
        axes[0, 1].text(0.5, 0.5, 'No Accuracy Data', ha='center', va='center', transform=axes[0, 1].transAxes)

    # 그래프 3: COCO AP (Average Precision)
    if 'coco/AP' in df.columns:
        axes[1, 0].plot(df['step'], df['coco/AP'], color='tab:green', label='AP', linewidth=2, marker='o', markersize=4)
        axes[1, 0].set_title('COCO AP Trend (Overall Precision)')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('AP')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        axes[1, 0].set_ylim(0, 1)
    else:
        axes[1, 0].text(0.5, 0.5, 'No COCO/AP Data', ha='center', va='center', transform=axes[1, 0].transAxes)

    # 그래프 4: COCO AR (Average Recall) 및 다른 지표들
    ax_ar = axes[1, 1]
    if 'coco/AR' in df.columns:
        ax_ar.plot(df['step'], df['coco/AR'], color='tab:orange', label='AR', linewidth=2, marker='s', markersize=4)
        ax_ar.set_title('COCO Metrics (AR, AP.5, AP.75)')
        ax_ar.set_xlabel('Step')
        ax_ar.set_ylabel('Score')
        ax_ar.set_ylim(0, 1)

        if 'coco/AP .5' in df.columns:
            ax_ar.plot(df['step'], df['coco/AP .5'], color='tab:purple', label='AP@0.5', linewidth=1.5, linestyle='--', alpha=0.7)
        if 'coco/AP .75' in df.columns:
            ax_ar.plot(df['step'], df['coco/AP .75'], color='tab:brown', label='AP@0.75', linewidth=1.5, linestyle='--', alpha=0.7)

        ax_ar.grid(True, alpha=0.3)
        ax_ar.legend()
    else:
        ax_ar.text(0.5, 0.5, 'No COCO/AR Data', ha='center', va='center', transform=ax_ar.transAxes)

    # 4. 이미지 저장
    plt.tight_layout(rect=[0, 0.03, 1, 0.9])
    plt.savefig(save_name, dpi=200)
    plt.close()
    
    print(f"✅ 그래프 저장 완료: {os.path.abspath(save_name)}")

# --- 실행부 ---
# 'train_log.json' 자리에 실제 파일 경로를 넣으세요.
# 예제 데이터를 위해 임시로 파일을 만들거나 기존 파일을 지정하시면 됩니다.
if __name__ == "__main__":
    # 파일 경로 지정
    #LOG_FILE_PATH = "/workspace/MMA/pose_log/mma_rtm_v1/20260106_084049/vis_data/20260106_084049.json" 
    LOG_FILE_PATH = "/workspace/MMA/pose_log/mma_rtm_v2/20260107_033218/vis_data/20260107_033218.json" 
    
    if os.path.exists(LOG_FILE_PATH):
        load_and_plot_logs(LOG_FILE_PATH, "learning_curve.png")
    else:
        print(f"❌ '{LOG_FILE_PATH}' 파일을 찾을 수 없습니다.")