import numpy as np
from numba import njit

import os
import re
import bottleneck as bn

from matplotlib import pyplot as plt
import matplotlib as mpl


class epix :
    """ 
        Python Script to 
    """

    def __init__(self,
                 fname: str,
                 nhead: int = 20,
                 nframe: int = 274996 // 2,
                 nrow: int = 176,
                 ncolumn: int = 192,
                 frequency: float = 1.0 ,
                 mean_gain: float=17.0
                 ) -> None:
        
        #Basic parameters
        self.mean_gain=mean_gain
        self.fname = fname
        self.nhead = nhead
        self.nframe = nframe
        self.nrow = nrow
        self.ncolumn = ncolumn
        self.npix = 4 * nrow * ncolumn

        self.frequency = float(frequency)     
        self.period_s  = 1.0 / self.frequency


        #Data Buffer
        self._buf = None          
        self._data = None         
        self._nframes = None
        self._load_data()

        # Timestamp
        self._t0 = self._parse_timestamp()

    
    def _parse_timestamp(self) -> np.datetime64:
        """
        Get the initial timestamp from the filename
        """
        m = re.search(r"(\d{8})_(\d{6})", os.path.basename(self.fname))
        if not m:
            return None
        date_part, time_part = m.groups()
        iso = (f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}T"
               f"{time_part[:2]}:{time_part[2:4]}:{time_part[4:]}")
        return np.datetime64(iso, "s")

    def _load_data(self) -> None:
        """
        mmep
        """
        if self._data is not None:
            return

        byte_size = os.path.getsize(self.fname)
        u16_size = byte_size // 2
        if u16_size % self.nframe:
            raise RuntimeError("File Length incorrect")

        self._nframes = u16_size // self.nframe
        self._buf = np.memmap(self.fname, mode="r", dtype=np.uint16)

        blk = self._buf.reshape(self._nframes, self.nframe)
        body = blk[:, self.nhead : self.nhead + self.npix]
        self._data = body.reshape(self._nframes,
                                  self.nrow,
                                  4 * self.ncolumn)


    # it is a property,
    @property
    def data(self) -> np.ndarray:
        self._load_data()
        return self._data

    @property
    def nframes(self) -> int:
        self._load_data()
        return self._nframes

    def frame(
        self,
        i: int,
        n: int = 1 
        ) -> np.ndarray:
        if n == 1:
            out = self.data[i]                  
        else:
            out = self.data[i:i + n]           
        return out


    
    # Time array 
    def _time_axis(self):
        N = self.nframes
        if self._t0 is None:
            return np.arange(N, dtype=float) * self.period_s

        if self.period_s.is_integer():
            step = np.timedelta64(int(self.period_s), "s")
            return np.arange(self._t0,
                             self._t0 + step * N,
                             step,
                             dtype="datetime64[s]")

        step_ns = int(round(self.period_s * 1e9))   
        base    = self._t0.astype("datetime64[ns]")
        return base + np.arange(N, dtype="timedelta64[ns]") * step_ns

    # Fast Data;
    def frame_mean(self) -> np.ndarray:
        return self.data.mean(axis=(1, 2))

    def time_series(self):
        return self._time_axis()

    def mean_series(self):
        return self.frame_mean()

    def pixel_series(self, x: int, y: int):
        return self.data[:, x, y]

    def pixel_series_p(self,x: int, y: int,p:int) : 
        return self.data[:,x,y+p*self.ncolumn]

    def low_occupy_dark(self):
        dark_2D = bn.nanmedian(self.data[0:2000], axis=0)
        return dark_2D

    def low_occupy_std(self):
        std_map = bn.nanstd(self.data[0:2000], axis=0)
        return std_map

    def get_dark(self):
        dark_2D = bn.nanmedian(self.data, axis=0)
        return dark_2D

    def get_std(self):
        std_map = bn.nanstd(self.data, axis=0)
        return std_map
    
    # Convert the image to show;
    @staticmethod
    def convert_image(image: np.ndarray) -> np.ndarray:
        row, col4 = image.shape
        col = col4 // 4
        im1 = image[:, 0*col : 1*col]
        im2 = image[:, 1*col : 2*col]
        im3 = image[::-1, 2*col : 3*col]   # flip 上半区
        im4 = image[::-1, 3*col : 4*col]
        return np.vstack([np.hstack([im3, im4]),
                          np.hstack([im1, im2])])


    # Common mode noise correction; 
    @staticmethod
    def _frame_common_mode(frame: np.ndarray, n0: int) -> None:
        idx = frame.ravel() < 4 * n0
        frame -= np.median(frame.ravel()[idx])

    @staticmethod
    def _col_common_mode(frame: np.ndarray, n1: int) -> None:
        ny, nx = frame.shape
        for icol in range(nx):
            col = frame[:, icol]
            idx = col < 4 * n1
            col -= np.median(col[idx])


    
    # ---------------------------------------------------------------------
    # Single frame processing without charging sharing correction 
    # ---------------------------------------------------------------------
    def _prep_frame(self,
                    iframe: int,
                    dark: np.ndarray,
                    n0: int,
                    n1: int,
                    col_cm: bool = True,
                    mask: np.ndarray | None = None,
                    gain: float | np.ndarray | None = None
                   ) -> np.ndarray:
        
        frame = self.frame(iframe).astype(float, copy=False)
        frame -= dark
        self._frame_common_mode(frame, n0)
        if col_cm:
            self._col_common_mode(frame, n1)

        if gain is None:
            frame /= self.mean_gain
        elif np.isscalar(gain):
            frame /= float(gain)
        else:
            if gain.shape != (self.nrow, 4 * self.ncolumn):
                raise ValueError(
                f"gain shape must be ({self.nrow}, {4*self.ncolumn}), "
                f"got {gain.shape}"
                )
            frame /= gain

        
        if mask is None:
            return frame
        elif mask.shape != (self.nrow, 4 * self.ncolumn):
            raise ValueError(
                f"mask shape must be ({self.nrow}, {4*self.ncolumn}),"
                f" got {mask.shape}"
            )
        else : 
            return frame*mask

    def background(self,
                   iframe: int,
                   dark: np.ndarray,
                   n0: int,
                   n1: int,
                   mask: np.ndarray | None = None,
                   gain: float | np.ndarray | None = None
                  ) -> np.ndarray:
        return self._prep_frame(iframe, dark, n0, n1, col_cm=True,mask=mask,gain=gain)

    def histogram(self,
                  iframe: int,
                  dark: np.ndarray,
                  n0: int,
                  n1: int,
                  bins,
                  mask: np.ndarray | None = None,
                  gain: float | np.ndarray | None = None) -> np.ndarray:
        frame = self._prep_frame(iframe, dark, n0, n1, col_cm=True,mask=mask,gain=gain)     
        return np.histogram(frame.ravel(), bins=bins)[0]

    def distribution(self,
                  iframe: int,
                  dark: np.ndarray,
                  n0: int,
                  n1: int,
                  bins,
                  mask: np.ndarray | None = None,
                  gain: float | np.ndarray | None = None) -> np.ndarray:
        frame = self._prep_frame(iframe, dark, n0, n1, col_cm=True,mask=mask,gain=gain)     
        return (frame>10).astype(int)
    
    def dis_histogram(self,
                  iframe: int,
                  dark: np.ndarray,
                  n0: int,
                  n1: int,
                  bins,
                  mask: np.ndarray | None = None,
                  gain: float | np.ndarray | None = None) -> np.ndarray:
        frame = self._prep_frame(iframe, dark, n0, n1, col_cm=True,mask=mask,gain=gain)  
        return np.histogram(frame.ravel(), bins=bins)[0],(frame>10).astype(int)
    
    # ---------------------------------------------------------------------
    # Charge-sharing (CS) Processing
    # ---------------------------------------------------------------------
    @staticmethod
    def _centroid_filter(frame: np.ndarray, n2: int) -> np.ndarray:
        ny, nx = frame.shape
        as1 = frame * (frame > 4 * n2)


        up = np.zeros_like(as1)
        down = np.zeros_like(as1)
        left = np.zeros_like(as1)
        right = np.zeros_like(as1)

        up[0:ny - 1, :] = as1[1:, :]
        down[1:, :] = as1[0:ny - 1, :]
        left[:, 0:nx - 1] = as1[:, 1:nx]
        right[:, 1:nx] = as1[:, 0:nx - 1]

        cen_v = as1 + up + down + left + right
        cen_f = (as1 > up) & (as1 > down) & (as1 > left) & (as1 > right)
        return cen_v * cen_f

    def cs_noise(self,
                 iframe: int,
                 dark: np.ndarray,
                 n0: int,
                 n1: int,
                 n2: int,
                 mask: np.ndarray | None = None,
                 gain: float | np.ndarray | None = None) -> np.ndarray:
        frame = self._prep_frame(iframe, dark, n0, n1, col_cm=True,mask=mask,gain=gain)
        return self._centroid_filter(frame, n2)

    def cs_histogram(self,
                     iframe: int,
                     dark: np.ndarray,
                     n0: int,
                     n1: int,
                     n2: int,
                     bins,
                     mask: np.ndarray | None = None,
                     gain: float | np.ndarray | None = None) -> np.ndarray:
        frame = self.cs_noise(iframe, dark, n0, n1, n2,mask=mask,gain=gain)
        return np.histogram(frame.ravel(), bins=bins)[0]


    def binary_scatter(raw_mask: np.ndarray,
                    *,
                    title: str = "Bad Pixels",
                    point_size: int = 3,
                    color: str = "red",
                    xlabel: str = "Column",
                    ylabel: str = "Row",
                    figsize=(7, 5),
                    invert_y: bool = True) -> None:

        mask= epix.convert_image(raw_mask)
        ny, nx = mask.shape
        Y, X = np.mgrid[0:ny, 0:nx]

        x_bad = X[mask].ravel()
        y_bad = Y[mask].ravel()
        print(f"bad pixels: {len(x_bad)}")

        plt.figure(figsize=figsize)
        plt.title(title)
        plt.scatter(x_bad, y_bad, s=point_size, color=color)

        if invert_y:
            plt.gca().invert_yaxis()
    
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.show()










