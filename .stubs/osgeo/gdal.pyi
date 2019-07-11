from typing import Optional

import numpy as np


class GDALDataType(int):
    ...


GDT_Byte: GDALDataType = ...
GDT_CFloat32: GDALDataType = ...
GDT_CFloat64: GDALDataType = ...
GDT_CInt16: GDALDataType = ...
GDT_CInt32: GDALDataType = ...
GDT_Float32: GDALDataType = ...
GDT_Float64: GDALDataType = ...
GDT_Int16: GDALDataType = ...
GDT_Int32: GDALDataType = ...
GDT_TypeCount: GDALDataType = ...
GDT_UInt16: GDALDataType = ...
GDT_UInt32: GDALDataType = ...
GDT_Unknown: GDALDataType = ...


class Band(object):
    def FlushCache(self) -> None:
        ...

    def ReadAsArray(
        self,
        xoff: int = 0,
        yoff: int = 0,
        win_xsize: Optional[int] = None,
        win_ysize: Optional[int] = None,
        buf_xsize: Optional[int] = None,
        buf_ysize: Optional[int] = None
    ) -> np.ndarray:
        ...

    def WriteArray(
        self, array, xoff: int = 0, yoff: int = 0, resample_alg: int = 0
    ) -> None:
        ...


class Dataset(object):
    def ReadAsArray(
        self,
        xoff: int = 0,
        yoff: int = 0,
        xsize: Optional[int] = None,
        ysize: Optional[int] = None
    ) -> np.ndarray:
        ...

    def FlushCache(self) -> None:
        ...

    def GetGeoTransform(self) -> str:
        ...

    def GetProjection(self) -> str:
        ...

    def GetRasterBand(self, band: int) -> Band:
        ...

    def SetGeoTransform(self, tran: str) -> None:
        ...

    def SetProjection(self, proj: str) -> None:
        ...


class Driver(object):
    def Create(
        self,
        path: str,
        xsize: int,
        ysize: int,
        bands: int = 1,
        etype: GDALDataType = GDT_Byte
    ) -> Dataset:
        ...


def Open(name: str) -> Dataset:
    ...


def GetDriverByName(name: str) -> Driver:
    ...
