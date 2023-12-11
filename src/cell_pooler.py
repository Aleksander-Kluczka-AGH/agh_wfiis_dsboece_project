import pandas as pd

from src.helper import expand_dataframe, explode_nw_data


class CellPooler:
    def __init__(self, raw_data):
        self._points = explode_nw_data(raw_data)
        self._points = expand_dataframe(self._points)

        # remove rows with cells which are not lte
        self._points = self._points[
            self._points["plmn lte"].apply(lambda x: len(x) > 0)
        ]

        # extract lte data from 'nw data' and convert str to correct types
        self._points["lte"] = self._points["nw data"].apply(lambda x: x.lte())
        self._points["lte"] = self._points["lte"].apply(self._to_dict)
        self._points["lte"] = self._points["lte"].apply(self._convert_types)
        self._points["time"] = self._points["time"].apply(pd.to_datetime)
        self._points[["longitude", "latitude"]] = self._points[
            ["longitude", "latitude"]
        ].apply(pd.to_numeric)

        # remove unnecessary columns
        self._points.drop("nw data", axis=1, inplace=True)
        self._points.drop("plmn wcdma", axis=1, inplace=True)

        # rename 'plmn lte' to 'bts_cell_id'
        self._points.rename(columns={"plmn lte": "bts_cell_id"}, inplace=True)

    @property
    def points(self):
        return self._points

    def _to_dict(self, lines) -> list[dict[str, str]]:
        def _convert_to_dict(raw_cell_info: str) -> dict[str, str]:
            retval: dict[str, str] = {}
            for pair in raw_cell_info:
                pair = pair.split("=")
                if len(pair) > 1:
                    retval[pair[0]] = pair[1]
            return retval

        modified_lines = []
        for line in lines:
            line.pop("type")

            line["signal"] = _convert_to_dict(line["signal"].split()[1:])
            line["identity"] = _convert_to_dict(line["identity"].split()[1:])

            modified_lines.append(line)

        return modified_lines

    def _convert_types(self, cell_infos: list):
        signal_columns = [
            "rsrp",
            "rsrq",
            "rssi",
            "rssnr",
            "ta",
            "cqi",
            "level",
            "parametersUseForLevel",
            "cqiTableIndex",
        ]
        identity_columns = [
            "mCi",
            "mPci",
            "mTac",
            "mEarfcn",
            "mBandwidth",
            "mMcc",
            "mMnc",
        ]
        for info in cell_infos:
            for column in signal_columns:
                try:
                    info["signal"][column] = pd.to_numeric(info["signal"][column])
                    if info["signal"][column] > 2147483640:
                        info["signal"][column] = float("inf")
                except KeyError:
                    pass

            for column in identity_columns:
                try:
                    info["identity"][column] = int(info["identity"][column])
                except KeyError:
                    pass
                except ValueError:
                    info["identity"][column] = None

        return cell_infos
