def xxyy2xywh(x1, y1, x2, y2):
    x = int((x1 + x2) / 2)
    y = int((y1 + y2) / 2)
    w = int(x2 - x1)
    h = int(y2 - y1)
    return x, y, w, h
