# adapted from https://github.com/sail-sg/zero-bubble-pipeline-parallelism
import json

import numpy as np
try:
    import drawsvg as draw
except ImportError:
    draw = None
import colorsys
import tempfile, os

def filter_time(e, start, end):
    # return e["start_time"] >= start and (end is None or e["completion_time"] <= end)
    # Check completion_time here to include the last long W
    return e.completion_time >= start and (end is None or e.completion_time <= end)


def load_json_data(filename, start=0, end=None, time_scale=1):
    with open(filename) as f:
        data = json.loads(f.read())
    fbw_types = {"F", "B", "W", "Optimizer"}
    return [[{
        "type": e["type"],
        "start_time": int(max(e["start_time"] - start, 0)) * time_scale,
        "completion_time": int(e["completion_time"] - start) * time_scale,
        "minibatch": e.get("minibatch", None),
    } for e in dev_evs
        if e["type"] in fbw_types and filter_time(e, start, end)
    ] for dev_evs in data]


ENABLE_BORDER = True
DISABLE_COL_BORDERS = True
ENABLE_BATCH_ID = True
ENABLE_EDGE_BLUR = False
SCALE_FACTOR = 4
S = SCALE_FACTOR

# TIME_PER_UNIT = 300 // SCALE_FACTOR
TIME_PER_UNIT = 4000 // SCALE_FACTOR


def to_color_fmt(c):
    # c = to_greyscale(c)
    return f"#{hex(c[0])[2:]}{hex(c[1])[2:]}{hex(c[2])[2:]}"


GREYSCALE_WEIGHTS = np.array([0.299, 0.587, 0.114])


def to_greyscale(color):
    c = np.dot(GREYSCALE_WEIGHTS, color[:3].astype(float)).astype(int)
    return np.array([c, c, c, 255])

# COLOR_MAP = {
#     "F": '#74AED4', # azure
#     "B": '#E63946', # Candy apple red
#     "D": '#ECA8A9', # 	Cornell Red
#     # "B": np.array([68, 211, 218]),  # sea color
#     # "W": to_color_fmt(np.array([47, 158, 73, 255])),
#     "W": '#D3E2B7', # Paris Green
#     # "W": np.array([224, 240, 231]),  # sea color
# }


COLOR_VALUE_MAP = {
    "F": np.array([114, 159, 220]), # azure
    "B": np.array([205, 92, 92]), # crimson glory
    "D": np.array([231, 143, 137]), # crimson
    # "B": np.array([68, 211, 218]),  # sea color
    # "W": to_color_fmt(np.array([47, 158, 73, 255])),
    "W": np.array([160, 218, 184]), # Paris Green
    # "W": np.array([224, 240, 231]),  # sea color
    # "Optimizer": to_color_fmt(np.array([255, 240, 197, 255])),
    "Optimizer": np.array([255, 217, 102]),
}
COLOR_MAP = {k: to_color_fmt(v) for k, v in COLOR_VALUE_MAP.items()}


# BORDER_SIZE = SCALE_FACTOR // 2
BORDER_SIZE = 0.15
SPAN_HEIGHT = SCALE_FACTOR * 10
FONT_SIZE = SCALE_FACTOR * 10
TITLE_WIDTH = SCALE_FACTOR * 25
CENTER_TITLE_HEIGHT = SPAN_HEIGHT * 6

WHITE = to_color_fmt(np.array([255, 255, 255, 255]))
BLACK = to_color_fmt(np.array([0, 0, 0, 255]))


class DrawCtx:
    def __init__(self, d, oy, ox):
        assert not isinstance(d, DrawCtx)
        self.d = d
        self.oy = oy
        self.ox = ox

    @classmethod
    def from_base_ctx(cls, base_ctx, oy, ox):
        assert isinstance(base_ctx, DrawCtx)
        return cls(base_ctx.d, base_ctx.oy + oy, base_ctx.ox + ox)

    def width(self):
        return self.d.width

    def height(self):
        return self.d.height

    def line(self, sy, sx, ey, ex, width=None):
        self.d.append(draw.Line(
            self.ox + sx,
            self.oy + sy,
            self.ox + ex,
            self.oy + ey,
            stroke='black',
            stroke_width=width or BORDER_SIZE,
        ))

    def rect(self, sy, sx, h, w, color):
        self.d.append(draw.Rectangle(
            self.ox + sx,
            self.oy + sy,
            w, h,
            fill=color,
            shape_rendering="geometricPrecision",
        ))

    def rect_frame(self, sy, sx, h, w):
        self.d.append(draw.Rectangle(
            self.ox + sx,
            self.oy + sy,
            w, h,
            fill="none",
            stroke=BLACK,
            stroke_width=BORDER_SIZE,
        ))

    def text(self, y, x, text, anchor="middle", font_scale=1, fill='black'):
        font_size = FONT_SIZE * font_scale
        tl = len(text) * font_size // 2
        self.d.append(draw.Text(
            text, font_size,
            self.ox + x,
            # Magic 3 to make it vertical center
            self.oy + y + font_size - 3,
            textLength=tl, lengthAdjust='spacing',
            text_anchor=anchor,
            font_family="Noto Sans",
            fill=fill,
            # font_style="oblique",
            # font_family="Computer Modern Roman",
        ))


def change_color_sat(c, percentage):
    c = c.astype(float) / 255.0
    (h, s, v) = colorsys.rgb_to_hsv(c[0], c[1], c[2])
    s *= percentage
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    c = np.array([r, g, b]) * 255
    return c.astype(int)


def draw_experiment_and_schedule(exp_events, sched_events, output_filename, tail=10):
    exp_canvas_info = CanvasInfo(exp_events, tail, 0)
    sched_canvas_info = CanvasInfo(sched_events, tail, 0, False)
    width = max(exp_canvas_info.get_canvas_size()[1], sched_canvas_info.get_canvas_size()[1])
    height = exp_canvas_info.get_canvas_size()[0] + sched_canvas_info.get_canvas_size()[0]

    include_w = True

    # d = draw.Drawing(width, sched_canvas_info.get_canvas_size()[0], origin="top-left")
    d = draw.Drawing(width, height, origin="top-left")
    ctx = DrawCtx(d, 0, 0)
    plot_events(ctx, sched_events, "", sched_canvas_info, include_w, include_o=False, include_info=False)
    # plot_events(ctx, sched_events, "", sched_canvas_info, include_w, include_o=False)
    # d.save_svg("pics/schedule.svg")

    # d = draw.Drawing(width, sched_canvas_info.get_canvas_size()[0], origin="top-left")
    # exp_ctx = DrawCtx(d, 0, 0)
    exp_ctx = DrawCtx.from_base_ctx(ctx, sched_canvas_info.get_canvas_size()[0], 0)
    plot_events(exp_ctx, exp_events, "", exp_canvas_info, include_w, include_o=True)
    # plot_events(exp_ctx, exp_events, "", exp_canvas_info, include_w, include_o=True)
    d.save_svg(output_filename)


def draw_events(events, output_filename, include_w=True, include_o=True, tail=50, longest_time=None, save=True, include_info=True):
    if draw is None:
        print("Warning: drawsvg is not installed, skipping SVG generation.")
        return None
    canvas_info = CanvasInfo(events, tail, center_title_height=0, enable_info=True, longest_time=longest_time)
    max_len = canvas_info.max_len
    # height = canvas_info.height
    # info_height = canvas_info.info_height
    height, width = canvas_info.get_canvas_size()

    d = draw.Drawing(width, height, origin="top-left")
    ctx = DrawCtx(d, 0, 0)

    plot_events(ctx, events, "", canvas_info, include_w, include_o, include_info)
    if save:
        d.save_svg(output_filename)
    return d


class CanvasInfo:
    def __init__(self, events, tail, center_title_height=CENTER_TITLE_HEIGHT, enable_info=True, longest_time=None):
        
        last_time = max(max([e["completion_time"] for e in dev_evs]) for dev_evs in events) if longest_time is None else longest_time
        self.max_len = (last_time + TIME_PER_UNIT - 1) // TIME_PER_UNIT + tail

        self.height = SPAN_HEIGHT * len(events) + BORDER_SIZE * (len(events) + 1)
        color_text_row_height = int(SPAN_HEIGHT * 1.6)
        self.color_text_height = color_text_row_height + BORDER_SIZE
        self.info_height = SPAN_HEIGHT + color_text_row_height + 3 * BORDER_SIZE
        if not enable_info:
            self.info_height /= 2
        self.center_title_height = center_title_height
        # self.center_title_height = 0

    def get_canvas_size(self):
        # height, width
        return self.height + self.info_height + self.center_title_height, self.max_len + TITLE_WIDTH


def plot_events(ctx, events, title_text: str, canvas_info: CanvasInfo, include_w=True, include_o=True, include_info=True):
    max_len = canvas_info.max_len
    height = canvas_info.height
    color_text_height = canvas_info.color_text_height
    info_height = canvas_info.info_height

    data_ctx = DrawCtx.from_base_ctx(ctx, 0, TITLE_WIDTH)

    for i, evs in enumerate(events):
        h = i * SPAN_HEIGHT + (i + 1) * BORDER_SIZE        
        wgrad_split = any([x["type"] == "W" for x in evs])
        # print(evs[0])
        for e in evs:
            if wgrad_split and e["type"] == "B":
                e["type"] = "D"
            start = BORDER_SIZE + e["start_time"] // TIME_PER_UNIT
            end = BORDER_SIZE + e["completion_time"] // TIME_PER_UNIT
            if start == end or not ENABLE_EDGE_BLUR:
                plot_span(data_ctx, start, end, h, COLOR_MAP[e["type"]])
            else:
                plot_span(data_ctx, start + 1, end - 1, h, COLOR_MAP[e["type"]])
                # plot_span(data_ctx, start, end - 1, h, COLOR_MAP[e["type"]])
                # c = change_color_sat(
                #     COLOR_VALUE_MAP[e["type"]],
                #     (e["start_time"] / TIME_PER_UNIT) % 1.0)
                # plot_span(data_ctx, start, start + 1, h, to_color_fmt(c))
                # c = change_color_sat(
                #     COLOR_VALUE_MAP[e["type"]],
                #     (e["completion_time"] / TIME_PER_UNIT) % 1.0)
                # plot_span(data_ctx, end - 1, end, h, to_color_fmt(c))

            if ENABLE_BATCH_ID and include_info:
                minibatch = str(e["minibatch"])
                center = (start + end) / 2.
                data_ctx.text(h+SPAN_HEIGHT / 4., center, minibatch, font_scale=0.6, fill='black' if e["chunk"] == 0 else 'white')
        if ENABLE_BORDER:
            data_ctx.line(h+SPAN_HEIGHT, 0, h+SPAN_HEIGHT+BORDER_SIZE, max_len - 1)

    if ENABLE_BORDER and not DISABLE_COL_BORDERS:
        data_ctx.line(0, 0, 0, max_len - 1)
        data_ctx.line(0, 0, height, 0)
        data_ctx.line(0, max_len - 1, height, max_len - 1)


    dev_title_ctx = DrawCtx.from_base_ctx(ctx, 0, 0)
    ndev = len(events)
    add_devices(dev_title_ctx, ndev)
    
    if not include_info:
        return

    # info_height = ndev * SPAN_HEIGHT + (ndev + 1) * BORDER_SIZE
    # info_ctx = DrawCtx.from_base_ctx(ctx, info_height, 0)
    # add_info(info_ctx, color_text_height, include_w, include_o)

    # if title_text:
    #     center_title_ctx = DrawCtx.from_base_ctx(info_ctx, canvas_info.info_height, 0)
    #     add_center_title(center_title_ctx, title_text)


def plot_span(ctx, start, end, h, color, ):
    ctx.rect(h, start, SPAN_HEIGHT, end - start, color)
    if ENABLE_BORDER:
        ctx.rect_frame(h-BORDER_SIZE, start, SPAN_HEIGHT + BORDER_SIZE, end - start)


def add_devices(ctx, devs):
    for i in range(devs):
        h = i * SPAN_HEIGHT + (i + 1) * BORDER_SIZE
        ctx.text(h, 6 * SCALE_FACTOR, "P{}".format(i), "left")


def add_info(ctx, color_text_height, include_w=True, include_o=True):
    div = 4 + int(include_w) + int(include_o)
    f_start = ctx.width() // div
    b_start = ctx.width() // div * 2
    w_start = ctx.width() // div * 3
    o_start = ctx.width() // div * 4

    block_w = 25 * SCALE_FACTOR
    plot_span(ctx, f_start, f_start+block_w, color_text_height + BORDER_SIZE, COLOR_MAP["F"])
    plot_span(ctx, b_start, b_start+block_w, color_text_height + BORDER_SIZE, COLOR_MAP["B"])
    if include_w:
        plot_span(ctx, w_start, w_start+block_w, color_text_height + BORDER_SIZE, COLOR_MAP["W"])
    if include_o:
        plot_span(ctx, o_start, o_start+block_w, color_text_height + BORDER_SIZE, COLOR_MAP["Optimizer"])

    ctx.text(0, 6 * SCALE_FACTOR, "Time", "left")
    draw_arrow(ctx, SPAN_HEIGHT // 2 + BORDER_SIZE + 1, 65 * SCALE_FACTOR, 50 * SCALE_FACTOR)

    block_w = 30 * SCALE_FACTOR
    ctx.text(color_text_height, f_start + block_w, "F", "left")
    ctx.text(color_text_height, b_start + block_w,
             "B", "left")
    if include_w:
        ctx.text(color_text_height, w_start + block_w, "W", "left")
    if include_o:
        ctx.text(color_text_height, o_start + block_w, "Optimizer Step", "left")


def add_center_title(ctx: DrawCtx, text):
    ctx.text(CENTER_TITLE_HEIGHT / 4, ctx.width() / 2,
             text, "middle", 2)


def draw_arrow(ctx: DrawCtx, start_y, start_x, width, thickness=2):
    b = thickness * (SCALE_FACTOR // 2)
    ctx.line(start_y, start_x, start_y, start_x + width, b)
    ctx.line(start_y, start_x + width, start_y - 3*b, start_x + width - 3*b)
    ctx.line(start_y, start_x + width, start_y + 3*b, start_x + width - 3*b)


def render_manual_graph(data, longest_time, enable_batch_id = False):
    global ENABLE_BORDER
    global ENABLE_BATCH_ID
    ENABLE_BORDER = True
    ENABLE_BATCH_ID = enable_batch_id
    fbw_types = {"F", "B", "W", "Optimizer"}
    start = 0
    end = None
    time_scale= 1024 / longest_time * TIME_PER_UNIT
    events = [[{
        "type": e.type,
        "start_time": int(max(e.start_time - start, 0)) * time_scale,
        "completion_time": int(e.completion_time - start) * time_scale,
        "minibatch": e.minibatch,
        "chunk": e.chunk if hasattr(e, "chunk") else 0,
    } for e in dev_evs
        if e.type in fbw_types and filter_time(e, start, end)
    ] for dev_evs in data]
    # events = load_json_data("std-schedule.json")
    # global TIME_PER_UNIT
    # global ENABLE_BATCH_ID
    # global ENABLE_BORDER
    # global SCALE_FACTOR
    # SCALE_FACTOR = 8
    # ENABLE_BATCH_ID = False
    # ENABLE_BORDER = False
    # TIME_PER_UNIT *= 7
    #events = load_json_data("no-bb-schedule.json")
    
    path = os.path.join(tempfile.mkdtemp(), 'a.svg')
    draw_events(events, path, include_w=True, include_o=False, tail=50, longest_time=longest_time * time_scale)
    return path


def render_experiment_graph():
    global ENABLE_BORDER
    global ENABLE_BATCH_ID
    global TIME_PER_UNIT
    ENABLE_BORDER = False
    ENABLE_BATCH_ID = False
    TIME_PER_UNIT = 200 // SCALE_FACTOR
    TIME_PER_UNIT *= 12000
    start_time = 1100000000 + 10000000
    # iter_time = 1600000000
    iter_time = 1290000000
    end_time = start_time + iter_time
    exp_events = load_json_data("20-09-zero/zero-events.json", start_time, end_time)
    # draw_events(events, "pics/experiment.svg")
    sched_events = load_json_data("schedule.json", time_scale=1000)
    draw_experiment_and_schedule(exp_events, sched_events, "pics/exp.svg")
    # draw_events(events, "pics/schedule.svg", include_w=True, include_o=False)


# render_manual_graph()
# render_experiment_graph()
