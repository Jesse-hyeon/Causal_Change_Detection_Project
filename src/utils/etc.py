import numpy as np
import matplotlib.pyplot as plt
import torch

### Masking 함수 정의
class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask

### Learning_rate 학습 함수 정의
def adjust_learning_rate(optimizer, epoch, args):
    if args["lradj"] == 'type1':
        lr_adjust = {epoch: args["learning_rate"] * (0.5 ** ((epoch - 1) // 1))}
    elif args["lradj"] == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args["lradj"] == 'type3':
        # 매 5 epoch마다 0.8배씩 줄이기
        base_lr = args["learning_rate"]
        decay_rate = 0.8
        step_size = 5
        lr = base_lr * (decay_rate ** (epoch // step_size))
        # 최소 lr 보장
        lr = max(lr, 1e-5)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f'Updating learning rate to {lr:.8f}')
        return  # type3은 여기서 끝나기 때문에 아래 if문은 건너뜀

    # type1, type2 처리
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


### EarlyStopping 함수 정의
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss

### 시각화 함수 정의
def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


### 평가지표 함수 정의
def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def R2(pred, true):
    pred = pred.flatten()
    true = true.flatten()
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    return 1 - (ss_res / ss_tot)


def ADJ_R2(pred, true, num_features):
    pred = pred.flatten()
    true = true.flatten()
    n = len(true)
    r2 = R2(pred, true)

    denom = n - num_features - 1
    if denom <= 0:
        print(f"[경고] Adjusted R² 계산 실패: n({n}) - num_features({num_features}) - 1 <= 0")
        return np.nan
    return 1 - (1 - r2) * (n - 1) / denom

def D_STAT(pred, true):
    """
    Compute directional accuracy D_stat (%)
    which measures how often the predicted direction matches the actual direction.
    """
    true = np.array(true)  # shape: (1, 1, 90, 1)
    pred = np.array(pred)

    # reshape to (batch, seq)
    true = true.reshape(true.shape[0], -1) # → (1, 90)
    pred = pred.reshape(pred.shape[0], -1)  # → (1, 90)

    true_diff = np.diff(true, axis=1)       # → (1, 89)
    pred_diff = np.diff(pred, axis=1)       # → (1, 89)

    correct_direction = (true_diff * pred_diff) >= 0
    return np.mean(correct_direction) * 100


def metric(pred, true, num_features):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    r2 =  R2(pred, true)
    adj_r2 = ADJ_R2(pred, true, num_features)
    D_stat = D_STAT(pred, true)

    return mae, mse, rmse, mape, mspe, r2, adj_r2, D_stat
