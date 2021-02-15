def contrastive_loss(ys, preds, margin=1):
    pred_sq = K.square(preds)
    margin_sq = K.square(K.maximum(margin - preds, 0))
    return K.mean(ys*pred_sq + (1 - ys)*margin_sq)


# Wrapped version
def contrastive_loss_with_margin(margin=1):
    def contrastive_loss(ys, preds):
        pred_sq = K.squrare(preds)
        margin_sq = K.square(K.maximum(margin - preds, 0))
        return K.mean(ys*pred_sq + (1 - ys)*margin_sq)
    return contrastive_loss

# ...
mod.compile(loss=contrastive_loss_with_margin(margin=1), optimzer=rms)


# Class version
class ContrastiveLoss(Loss):
    def __init__(self, margin):
        super().__init__()
        self.margin = margin

    def call(self, ys, preds):
        pred_sq = K.squrare(preds)
        margin_sq = K.square(K.maximum(self.margin - preds, 0))
        return K.mean(ys*pred_sq + (1 - ys)*margin_sq)

# ...
mod.compile(loss=ContrastiveLoss(margin=1), optimizer=rms)
