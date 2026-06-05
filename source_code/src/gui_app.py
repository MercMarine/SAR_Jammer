"""

Author - MercMarine
GitHub - https://github.com/MercMarine

gui_app.py - Отвечает за графический интерфейс.

"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image
import datetime
import zipfile
import os
import struct
import gc
import sys
import atexit
import threading

Image.MAX_IMAGE_PIXELS = 1_000_000_000 # Максимальный размер РГГ (H x W)

from signals import SinusoidalSignal, LCHMSignal
from image_processor import ImageLoader

MAX_GUI_DIM = 2000 # Ограничение размеров окна отрисовки РГГ

# Словарь-сопоставитель при загрузке параметров

SIGNAL_TYPE_MAP = {
    "Sinusoidal": "Синусоидальный", "Chirp": "ЛЧМ",
    "Синусоидальный": "Синусоидальный", "ЛЧМ": "ЛЧМ"
}

# Диапазон параметров

PARAM_RANGES = {
    "distance": (400, 1200), "power_kw": (0.1, 10.0),
    "freq": (-200, 200), "bw": (10, 200), "tau_jammer": (0.2, 10.0), #freq 5,200
    "T_j_pri": (1.0, 20.0), "T_p": (1.0, 15.0), "tau_rec": (0.2, 10.0),
    "gain": (1.0, 1000.0), "P_ch": (0.0, 0.5)
}

class SARJammingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SAR Jammer v1.3")
        self.root.geometry("1400x820")

        # Синтетическое РГГ
        H, W = 512, 512
        mag = np.random.rand(H, W) * 0.3 + 0.4
        phase = np.random.uniform(0, 2 * np.pi, (H, W))

        self.image_full = mag * np.exp(1j * phase)
        self.image_full_memmap = None
        self.is_bin_source = False

        self.image = self.image_full.copy()
        self.image_with_jamming = self.image.copy()
        self.display_res_label = tk.StringVar(value="Разрешение: 512x512")
        self.file_path = None

        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        atexit.register(self._cleanup)

        self._create_widgets()
        self._setup_layout()
        self.update_plots()

        self.root.state('zoomed')
        try:
            self.root.iconbitmap('app_icon.ico')
        except Exception:
            pass
        self.root.bind("<Escape>", lambda e: self.root.attributes('-fullscreen', False))

    def _on_closing(self):
        self._cleanup()
        self.root.destroy()
        sys.exit(0)

    def _cleanup(self):
        if hasattr(self, 'image_full_memmap') and self.image_full_memmap is not None:
            del self.image_full_memmap
            self.image_full_memmap = None
        gc.collect()
        plt.close('all')

    def _create_widgets(self):
        ctrl_frame = ttk.Frame(self.root, padding=10)
        self.ctrl_frame = ctrl_frame

        ttk.Label(ctrl_frame, text="ПАРАМЕТРЫ ППР", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(5,0))
        self.signal_type = ttk.Combobox(ctrl_frame, values=["Синусоидальный", "ЛЧМ"], state="readonly")
        self.signal_type.current(0)
        self.signal_type.pack(fill="x", pady=5)
        self.signal_type.bind("<<ComboboxSelected>>", lambda e: self._on_param_change())
        self.var_dist, _ = self._add_slider_with_entry(ctrl_frame, "Дальность до ППР (км):", 400, 1200, 600, 10, self._on_param_change)
        self.var_power, _ = self._add_slider_with_entry(ctrl_frame, "Мощность ППР (кВт):", 0.1, 10.0, 1.0, 0.1, self._on_param_change)
        self.var_freq, _ = self._add_slider_with_entry(ctrl_frame, "Несущая частота ППР (МГц):", -200, 200, 50, 1, self._on_param_change)
        self.var_bw, _ = self._add_slider_with_entry(ctrl_frame, "Полоса ЛЧМ ППР (МГц):", 10, 200, 80, 1, self._on_param_change)
        self.var_tau_j, _ = self._add_slider_with_entry(ctrl_frame, "Длительность импульса ППР tau (мкс):", 0.2, 10.0, 2.0, 0.1, self._on_param_change)
        self.var_Tj_pri, _ = self._add_slider_with_entry(ctrl_frame, "Период повторения ППР T_ppr (мкс):", 1.0, 20.0, 5.0, 0.1, self._on_param_change)
        self.var_azimuth, self.scale_azimuth = self._add_slider_with_entry(ctrl_frame, "Номер столбца появления помехи:", 50, 1200, 250, 1, self._on_param_change)
        self.var_width, self.scale_width = self._add_slider_with_entry(ctrl_frame, "Количество занимаемых столбцов:", 10, 1000, 50, 1, self._on_param_change)
        self.var_gain, _ = self._add_slider_with_entry(ctrl_frame, "Коэфф. усиления сигнала:", 1.0, 1000.0, 100.0, 10.0, self._on_param_change)

        ttk.Label(ctrl_frame, text="ПАРАМЕТРЫ ЗАПИСИ/ПРИЕМА РСА", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(15,0))
        self.var_Tp, _ = self._add_slider_with_entry(ctrl_frame, "Период записи T_p (мкс):", 1.0, 15.0, 5.0, 0.1, self._on_param_change)
        self.var_tau_rec, _ = self._add_slider_with_entry(ctrl_frame, "Длительность записи tau_s (мкс):", 0.2, 10.0, 2.0, 0.1, self._on_param_change)

        ttk.Label(ctrl_frame, text="ВИЗУАЛИЗАЦИЯ И ПОРОГ P_ch", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(15,0))
        ttk.Label(ctrl_frame, text="Режим отображения РГГ:").pack(anchor="w")
        self.display_mode = ttk.Combobox(ctrl_frame, values=["|S| (Модуль сигнала)", "|Re| (Действительная)"], state="readonly")
        self.display_mode.current(0)
        self.display_mode.pack(fill="x", pady=5)
        self.display_mode.bind("<<ComboboxSelected>>", lambda e: self._on_param_change())
        self.var_pch, _ = self._add_slider_with_entry(ctrl_frame, "Чувствительность P_ch (порог отсечки):", 0.0, 0.5, 0.0, 0.01, self._on_param_change)

        ttk.Label(ctrl_frame, textvariable=self.display_res_label, font=("Segoe UI", 9), foreground="gray").pack(anchor="w", pady=5)
        btn_frame = ttk.Frame(ctrl_frame)
        btn_frame.pack(fill="x", pady=10)
        ttk.Button(btn_frame, text="Загрузить РГГ или .bin", command=self.load_image).pack(side="left", fill="x", expand=True, padx=2)
        ttk.Button(btn_frame, text="Сбросить параметры", command=self._reset_params).pack(side="right", fill="x", expand=True, padx=2)
        file_btn_frame = ttk.Frame(ctrl_frame)
        file_btn_frame.pack(fill="x", pady=5)
        ttk.Button(file_btn_frame, text="Сохранить в ZIP", command=self.save_results).pack(side="left", fill="x", expand=True, padx=2)
        ttk.Button(file_btn_frame, text="Загрузить параметры", command=self.load_parameters).pack(side="right", fill="x", expand=True, padx=2)

        plot_frame = ttk.Frame(self.root, padding=10)
        self.plot_frame = plot_frame
        self.fig, self.axes = plt.subplots(2, 3, figsize=(14, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    # Слайдер
    def _add_slider_with_entry(self, parent, label, from_, to, initial, resolution, command):
        var = tk.DoubleVar(value=initial)
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=4)
        ttk.Label(frame, text=label, anchor="w").pack(fill="x")
        sub_frame = ttk.Frame(frame)
        sub_frame.pack(fill="x")
        scale = ttk.Scale(sub_frame, from_=from_, to=to, variable=var, orient="horizontal")
        scale.pack(side="left", fill="x", expand=True, padx=(0, 8))
        entry = ttk.Entry(sub_frame, textvariable=var, width=8, justify="center")
        entry.pack(side="right")

        def validate_entry(*args):
            try:
                val = float(var.get())
                if val < from_: val = from_
                elif val > to: val = to
                var.set(round(val, 3))
                command()
            except (ValueError, tk.TclError):
                pass

        entry.bind("<FocusOut>", lambda e: validate_entry())
        entry.bind("<Return>", lambda e: validate_entry())
        scale.config(command=lambda e: command())
        return var, scale

    def _setup_layout(self):
        self.ctrl_frame.pack(side="left", fill="y", padx=10, pady=10)
        self.plot_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

    # Загрузка файла
    def load_image(self):
        path = filedialog.askopenfilename(
            title="Загрузить РГГ",
            filetypes=[("All supported", "*.png *.jpg *.jpeg *.bmp *.tiff *.bin"),
                       ("Images", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                       ("Binary Hologram", "*.bin")]
        )
        if not path:
            return

        progress_window = tk.Toplevel(self.root)
        progress_window.title("Загрузка...")
        progress_window.geometry("400x120")
        progress_window.transient(self.root)
        progress_window.grab_set()
        progress_label = ttk.Label(progress_window, text="Инициализация...", font=("Segoe UI", 10))
        progress_label.pack(pady=10)
        progress_bar = ttk.Progressbar(progress_window, mode='indeterminate', length=350)
        progress_bar.pack(pady=10)
        progress_bar.start(10)

        def load_in_thread():
            try:
                if path.lower().endswith('.bin'):
                    result = self._load_bin_hologram_optimized(path, progress_label)
                else:
                    result = self._load_pil_image_optimized(path, progress_label)

                self.file_path = path
                self.root.after(0, lambda: self._finalize_load(result, progress_window))
            except Exception as e:
                progress_window.destroy()
                self.root.after(0, lambda: messagebox.showerror("Ошибка загрузки", str(e)))

        threading.Thread(target=load_in_thread, daemon=True).start()

    def _load_bin_hologram_optimized(self, path, progress_label):
        progress_label.config(text="Чтение заголовка...")
        file_size = os.path.getsize(path)
        with open(path, 'rb') as f:
            header = f.read(8)
            M_f, N_f = struct.unpack('<ff', header)
            h, w = int(round(M_f)), int(round(N_f))

        # Ограничения на размеры РГГ
        if h <= 0 or w <= 0 or h > 100000 or w > 100000:
            raise ValueError(f"Некорректные размеры: {w}x{h}")

        # Проверка на корректность размерности
        expected_size = 8 + h * w * 8
        if file_size < expected_size:
            raise ValueError(f"Размер файла ({file_size}) меньше ожидаемого ({expected_size})")

        progress_label.config(text=f"Создание memmap ({w}x{h})...")
        raw_memmap = np.memmap(path, dtype='<c8', mode='r', offset=8, shape=(h, w), order='F')
        progress_label.config(text="Генерация превью...")
        gui_image = self._auto_downscale_view(raw_memmap)

        return {
            'full': raw_memmap,
            'memmap': raw_memmap,
            'gui': gui_image,
            'is_bin': True,
            'shape': (h, w)
        }

    def _load_pil_image_optimized(self, path, progress_label):
        progress_label.config(text="Загрузка изображения...")
        img_pil = Image.open(path).convert('L')
        W_orig, H_orig = img_pil.size
        mag = np.array(img_pil, dtype=np.float32) / 255.0
        phase = np.random.uniform(0, 2 * np.pi, mag.shape).astype(np.float32)

        if W_orig * H_orig > 50_000_000:
            progress_label.config(text="Конвертация в complex64 (чанками)...")
            full_complex = np.empty(mag.shape, dtype=np.complex64)
            full_complex.real = mag * np.cos(phase)
            full_complex.imag = mag * np.sin(phase)
        else:
            full_complex = mag.astype(np.complex64) * np.exp(1j * phase).astype(np.complex64)

        progress_label.config(text="Подготовка данных...")
        gui_image = self._auto_downscale_view(full_complex)
        return {
            'full': full_complex,
            'memmap': None,
            'gui': gui_image,
            'is_bin': False,
            'shape': full_complex.shape
        }

    def _finalize_load(self, result, progress_window):
        progress_window.destroy()
        self.image_full = result['full']
        self.image_full_memmap = result['memmap']
        self.is_bin_source = result['is_bin']
        self.image = result['gui'].copy()
        self.image_with_jamming = self.image.copy()

        H, W = result['shape']
        mem_mb = (H * W * 8) / (1024 * 1024) if result['memmap'] is None else 0
        mem_text = f" (memmap, ~0 МБ в RAM)" if result['memmap'] is not None else f" ({mem_mb:.1f} МБ)"

        self.display_res_label.set(f"Разрешение: {W}x{H}{mem_text}")
        self._update_sliders_for_resolution(W, H)
        self.update_plots()

    # Генерация помехового сигнала

    def _generate_jammed_complex(self, img_complex, params, spatial_scale=1.0):
        H, W = img_complex.shape
        TIME_WINDOW_PER_ROW_US = 20.0
        sample_rate = H / TIME_WINDOW_PER_ROW_US  # Дискретизация привязана к высоте столбца
        col_center = int(params["azimuth"] * spatial_scale)
        col_width = int(params["width"] * spatial_scale)

        start_col = max(0, col_center - col_width // 2)
        end_col = min(W, col_center + col_width // 2)
        actual_w = end_col - start_col
        if actual_w <= 0: actual_w = 1

        total_len = H * actual_w
        time_axis_cont = np.arange(total_len) / sample_rate

        sig_kwargs = {"amplitude": 1.0, "length": total_len, "sample_rate": sample_rate}
        if params["signal_type"] == "Синусоидальный":
            signal = SinusoidalSignal(frequency=params["freq"], **sig_kwargs)
        else:
            signal = LCHMSignal(f_center=params["freq"], bandwidth=params["bw"], **sig_kwargs)
        signal.generate()
        raw_signal_1d = signal.signal

        jammer_mask_1d = (np.mod(time_axis_cont, params["T_j_pri"]) <= params["tau_jammer"]).astype(float) if params[
                                                                                                                  "T_j_pri"] > 0 else np.ones_like(
            raw_signal_1d.real)
        sar_mask_1d = (np.mod(time_axis_cont, params["T_p"]) <= params["tau_rec"]).astype(float) if params[
                                                                                                        "T_p"] > 0 else np.ones_like(
            raw_signal_1d.real)

        received_1d = raw_signal_1d * jammer_mask_1d * sar_mask_1d

        jamming_2d = received_1d.reshape(H, actual_w, order='F')

        base_intensity = (params["power_kw"] * params["gain"]) / 100.0
        azimuth_coords = np.linspace(-1.0, 1.0, actual_w)
        sinc_env = np.clip(np.sinc(azimuth_coords), 0, None)
        interference_2d = jamming_2d * sinc_env[np.newaxis, :]

        jamming_patch = base_intensity * interference_2d
        if params["P_ch"] > 0:
            jamming_patch[np.abs(jamming_patch) < params["P_ch"]] = 0

        img_out = img_complex.copy()
        img_out[:, start_col:end_col] += jamming_patch
        return img_out

    # Сохранение результатов

    def save_results(self):
        directory = filedialog.askdirectory(title="Выберите папку для сохранения архива")
        if not directory: return

        progress_window = tk.Toplevel(self.root)
        progress_window.title("Сохранение...")
        progress_window.geometry("400x150")
        progress_window.transient(self.root)
        progress_window.grab_set()

        progress_label = ttk.Label(progress_window, text="Инициализация...", font=("Segoe UI", 10))
        progress_label.pack(pady=10)
        progress_bar = ttk.Progressbar(progress_window, mode='determinate', length=350)
        progress_bar.pack(pady=10)
        cancel_var = tk.BooleanVar(value=False)
        ttk.Button(progress_window, text="Отмена", command=lambda: cancel_var.set(True)).pack(pady=5)

        def save_in_thread():
            try:
                date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                params = self._get_params()

                full_img = self.image_full_memmap if (
                            self.is_bin_source and self.image_full_memmap is not None) else self.image_full
                full_h, full_w = full_img.shape

                progress_label.config(text="Генерация помехи (0%)...")
                progress_bar['value'] = 0
                progress_window.update()

                gui_w = self.image.shape[1]
                scale_x = full_w / gui_w if gui_w > 0 else 1.0

                if full_h > 10000:
                    jammed_full = np.empty_like(full_img)
                    chunk_rows = 2000
                    for i in range(0, full_h, chunk_rows):
                        if cancel_var.get(): raise Exception("Отменено")
                        end_i = min(i + chunk_rows, full_h)
                        jammed_full[i:end_i] = self._generate_jammed_complex(full_img[i:end_i].copy(), params, scale_x)
                        progress_bar['value'] = int(25 * i / full_h)
                        progress_window.update()
                else:
                    jammed_full = self._generate_jammed_complex(full_img, params, scale_x)

                progress_bar['value'] = 25
                progress_window.update()

                # Скриншот и параметры
                progress_label.config(text="Сохранение графиков...")
                png_name = f"graphics_{full_h}_{full_w}_{date_str}.png"
                img_path = os.path.join(directory, png_name)
                self.fig.savefig(img_path, dpi=150, bbox_inches='tight')
                progress_bar['value'] = 50
                progress_window.update()

                txt_name = f"params_holo_{full_h}_{full_w}_{date_str}.txt"
                txt_path = os.path.join(directory, txt_name)
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write("# SAR Jamming Simulation Parameters\n")
                    f.write(f"# Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write(f"signal_type: {'Sinusoidal' if params['signal_type'] == 'Синусоидальный' else 'Chirp'}\n")
                    for key, value in params.items():
                        if key != "signal_type": f.write(f"{key}: {value}\n")

                progress_bar['value'] = 60
                progress_window.update()

                # Сохранение РГГ

                progress_label.config(text="Запись .bin файла (60-90%)...")
                bin_name = f"holo_{full_h}_{full_w}_{date_str}.bin"
                bin_path = os.path.join(directory, bin_name)

                chunk_cols = 1000
                with open(bin_path, 'wb') as bf:
                    bf.write(struct.pack('<ff', float(full_h), float(full_w)))

                    for j in range(0, full_w, chunk_cols):
                        if cancel_var.get(): raise Exception("Отменено")

                        chunk = jammed_full[:, j:j + chunk_cols].astype('<c8')

                        bf.write(chunk.tobytes(order='F'))

                        pct = 60 + int(30 * min(j + chunk_cols, full_w) / full_w)
                        progress_label.config(text=f"Запись .bin ({pct}%)...")
                        progress_bar['value'] = pct
                        progress_window.update()
                        del chunk
                        gc.collect()

                progress_bar['value'] = 95
                progress_window.update()

                # Упаковка в zipы
                progress_label.config(text="Создание ZIP архива...")
                zip_name = f"sar_results_{full_h}x{full_w}_{date_str}.zip"
                zip_path = os.path.join(directory, zip_name)
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as zipf:
                    zipf.write(img_path, png_name)
                    zipf.write(txt_path, txt_name)
                    zipf.write(bin_path, bin_name)

                os.remove(img_path);
                os.remove(txt_path);
                os.remove(bin_path)
                progress_bar['value'] = 100
                progress_label.config(text="✓ Готово!")
                progress_window.after(1000, progress_window.destroy)

                messagebox.showinfo("Успех", f"Архив сохранён:\n{zip_path}\nРазмер: {full_w}x{full_h}")

            except Exception as e:
                progress_window.destroy()
                if "Отменено" not in str(e):
                    messagebox.showerror("Ошибка", str(e))

        threading.Thread(target=save_in_thread, daemon=True).start()

    def load_parameters(self):
        file_path = filedialog.askopenfilename(title="Загрузить параметры", filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if not file_path: return

        param_map = {
            "signal_type": (self.signal_type, lambda x: x),
            "distance": (self.var_dist, float), "power_kw": (self.var_power, float),
            "freq": (self.var_freq, float), "bw": (self.var_bw, float),
            "tau_jammer": (self.var_tau_j, float), "T_j_pri": (self.var_Tj_pri, float),
            "azimuth": (self.var_azimuth, float), "width": (self.var_width, float),
            "gain": (self.var_gain, float), "T_p": (self.var_Tp, float),
            "tau_rec": (self.var_tau_rec, float), "display_mode": (self.display_mode, lambda x: x),
            "P_ch": (self.var_pch, float)
        }

        warnings = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith("#"): continue
                    if ":" in line:
                        key, value_str = line.split(":", 1)
                        key, value_str = key.strip(), value_str.strip()
                        if key in param_map:
                            var_widget, converter = param_map[key]
                            try:
                                val = converter(value_str)
                                if key in PARAM_RANGES:
                                    mn, mx = PARAM_RANGES[key]
                                    if val < mn or val > mx:
                                        warnings.append(f"'{key}'={val} вне [{mn},{mx}]. Установлено {max(mn, min(mx, val))}")
                                        val = max(mn, min(mx, val))
                                if key == "signal_type":
                                    if val in SIGNAL_TYPE_MAP: val = SIGNAL_TYPE_MAP[val]
                                    else: warnings.append(f"Неизвестный тип сигнала '{val}'. По умолчанию."); val = "Синусоидальный"
                                var_widget.set(val)
                            except (ValueError, tk.TclError):
                                warnings.append(f"Некорректное значение '{key}': '{value_str}'. По умолчанию.")
                        else:
                            warnings.append(f"Неизвестный параметр в строке {line_num}: '{key}'")
            if warnings: messagebox.showwarning("Предупреждения", "\n".join(warnings))
            self._on_param_change()
            messagebox.showinfo("Успех", "Параметры загружены.")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось прочитать файл:\n{e}")

    # Считывание параметров

    def _get_params(self):
        return {
            "signal_type": self.signal_type.get(), "distance": self.var_dist.get(),
            "power_kw": self.var_power.get(), "freq": self.var_freq.get(),
            "bw": self.var_bw.get(), "tau_jammer": self.var_tau_j.get(),
            "T_j_pri": self.var_Tj_pri.get(),
            "azimuth": int(self.var_azimuth.get()), "width": int(self.var_width.get()),
            "T_p": self.var_Tp.get(), "tau_rec": self.var_tau_rec.get(),
            "display_mode": self.display_mode.get(),
            "P_ch": self.var_pch.get(), "gain": self.var_gain.get()
        }

    def _on_param_change(self, event=None):
        if self.var_tau_j.get() > self.var_Tj_pri.get(): self.var_tau_j.set(self.var_Tj_pri.get())
        if self.var_tau_rec.get() > self.var_Tp.get(): self.var_tau_rec.set(self.var_Tp.get())
        H, W = self.image.shape
        if int(self.var_width.get()) > W // 2: self.var_width.set(W // 2)
        self.update_plots()

    @staticmethod
    def _normalize_for_display(img_complex, mode):
        if mode.startswith("|S|"):
            data = np.abs(img_complex)
            log_data = np.log1p(data)
            mn, mx = np.min(log_data), np.max(log_data)
            return (log_data - mn) / (mx - mn + 1e-9) if mx > mn else np.zeros_like(log_data)
        else:
            return np.real(img_complex)

    def update_plots(self):
        p = self._get_params()
        self.image_with_jamming = self._generate_jammed_complex(self.image, p, spatial_scale=1.0)

        self._clear_axes()
        disp_orig = self._normalize_for_display(self.image, p["display_mode"])
        disp_jam = self._normalize_for_display(self.image_with_jamming, p["display_mode"])

        is_re_mode = p["display_mode"].startswith("|Re|")

        if is_re_mode:
            max_abs = max(np.abs(disp_orig).max(), np.abs(disp_jam).max())
            if max_abs == 0: max_abs = 1.0

            vmin, vmax = -max_abs, max_abs
            cmap = 'gray'
        else:
            vmin, vmax = 0, 1
            cmap = 'gray'

        # Графики

        self.axes[0, 0].imshow(disp_orig, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
        self.axes[0, 0].set_title("Исходная РГГ");
        self.axes[0, 0].axis("off")
        self.axes[0, 1].imshow(disp_jam, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
        self.axes[0, 1].set_title("РГГ с помехой");
        self.axes[0, 1].axis("off")
        self.axes[0, 2].axis('off')

        H, W = self.image.shape
        TIME_WINDOW_PER_ROW_US = 20.0
        sample_rate = H / TIME_WINDOW_PER_ROW_US
        time_slice = np.arange(H) / sample_rate

        rec_slice = self.image_with_jamming[:, W // 2]
        raw_slice = self.image[:, W // 2]

        mode = p["display_mode"]
        if mode.startswith("|S|"):
            raw_plot = np.abs(raw_slice);
            rec_plot = np.abs(rec_slice)
            max_val = max(np.max(raw_plot), np.max(rec_plot))
            if max_val > 0: raw_plot /= max_val; rec_plot /= max_val
            y_lim, ylabel = (0, 1.1), "Нормированная амплитуда"
        else:
            raw_plot = np.real(raw_slice);
            rec_plot = np.real(rec_slice)
            max_val = max(np.max(np.abs(raw_plot)), np.max(np.abs(rec_plot)))
            if max_val > 0: raw_plot /= max_val; rec_plot /= max_val
            y_lim, ylabel = (-1.1, 1.1), "Нормированная амплитуда"

        self.axes[1, 0].plot(time_slice, raw_plot, 'b-', linewidth=1.5, alpha=0.8)
        self.axes[1, 0].set_ylim(y_lim);
        self.axes[1, 0].set_ylabel(ylabel);
        self.axes[1, 0].set_xlabel("Время, мкс")
        self.axes[1, 0].set_title("Исходный помеховый сигнал");
        self.axes[1, 0].grid(True, alpha=0.3)

        self.axes[1, 1].plot(time_slice, rec_plot, 'r-', linewidth=1.8)
        self.axes[1, 1].set_ylim(y_lim);
        self.axes[1, 1].set_ylabel(ylabel);
        self.axes[1, 1].set_xlabel("Время, мкс")
        self.axes[1, 1].set_title("Принятый помеховый сигнал");
        self.axes[1, 1].grid(True, alpha=0.3)

        fs_spec, spec_len = 500.0, 4096
        t_spec = np.arange(spec_len) / fs_spec
        if p["signal_type"] == "Синусоидальный":
            sig_for_spec = np.exp(1j * 2 * np.pi * p["freq"] * t_spec)
        else:
            f_start, f_end = p["freq"] - p["bw"] / 2, p["freq"] + p["bw"] / 2
            chirp_rate = (f_end - f_start) / t_spec[-1] if t_spec[-1] > 0 else 0
            sig_for_spec = np.exp(1j * 2 * np.pi * (f_start * t_spec + 0.5 * chirp_rate * t_spec ** 2))
        spectrum = np.fft.fftshift(np.abs(np.fft.fft(sig_for_spec)))
        freq_axis = np.fft.fftshift(np.fft.fftfreq(spec_len, 1 / fs_spec))
        self.axes[1, 2].plot(freq_axis, spectrum, 'purple', linewidth=1.5)
        self.axes[1, 2].set_title("Спектр помехового сигнала")
        self.axes[1, 2].set_xlabel("Частота, МГц");
        self.axes[1, 2].set_ylabel("|S(f)|")
        self.axes[1, 2].grid(True, alpha=0.3)

        self.fig.suptitle(f"T_ppr={p['T_j_pri']:.1f} мкс | T_p={p['T_p']:.1f} мкс | Усиление: x{p['gain']:.1f}",
                          fontsize=11, fontweight='bold', y=0.98)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    # Очистка данных графиков перед их обновлением

    def _clear_axes(self):
        for ax in self.axes.flat: ax.clear()

    # Сброс до параметров по умолчанию

    def _reset_params(self):
        self.var_dist.set(600); self.var_power.set(1.0); self.var_freq.set(50); self.var_bw.set(80)
        self.var_tau_j.set(2.0); self.var_Tj_pri.set(5.0)
        self.var_azimuth.set(250); self.var_width.set(50); self.var_gain.set(100.0)
        self.var_Tp.set(5.0); self.var_tau_rec.set(2.0)
        self.display_mode.current(0); self.var_pch.set(0.0)
        self.update_plots()

    # уменьшает отображение РГГ в окне пользователя

    def _auto_downscale_view(self, arr, max_dim=MAX_GUI_DIM):
        h, w = arr.shape
        if max(h, w) <= max_dim: return arr.copy()
        step_y = max(1, int(h / max_dim))
        step_x = max(1, int(w / max_dim))
        return arr[::step_y, ::step_x].copy()

    # Подстраивание пределов ползунков под разрешение

    def _update_sliders_for_resolution(self, w, h):
        self.scale_azimuth.configure(to=max(10, w-10))
        self.scale_azimuth.set(min(self.var_azimuth.get(), w-10))
        self.scale_width.configure(to=min(200, w//2))
        self.scale_width.set(min(self.var_width.get(), w//2))

if __name__ == "__main__":
    root = tk.Tk()
    app = SARJammingGUI(root)
    root.mainloop()