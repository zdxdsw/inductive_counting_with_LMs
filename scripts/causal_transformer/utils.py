def get_acc(logits, labels, ignore_index):
    pred = logits.argmax(dim=-1)

    counting_correct, counting_demo, last_correct, last_demo = 0, 0, 0, 0
    for _pred, _labels in zip(pred, labels):
        counting_pred, last_pred = _pred[_labels != ignore_index][:-1], _pred[_labels != ignore_index][-1].squeeze()
        counting_label, last_label = _labels[_labels != ignore_index][:-1], _labels[_labels != ignore_index][-1].squeeze()

        counting_correct += (counting_pred == counting_label).float().sum().item()
        counting_demo += counting_label.numel()
        last_correct += (last_pred == last_label).float().sum().item()
        last_demo += last_label.numel()
    return counting_correct, counting_demo, last_correct, last_demo