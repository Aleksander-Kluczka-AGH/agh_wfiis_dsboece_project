import logging as log

import pandas as pd

from src.helper import expand_dataframe, explode_nw_data


class MobileDataParser:
    def __init__(self, raw_data: pd.DataFrame) -> None:
        log.info("Created MobileDataParser, parsing raw data...")
        self.raw_data = self._prune_invalid_data(raw_data)
        self._measurement_points = self._parse_to_measurement_points(self.raw_data)

    def get_measurement_points(self) -> pd.DataFrame:
        return self._measurement_points

    def get_event_points(self, thresholds: dict[str, float] = {}) -> pd.DataFrame:
        # params to detect events
        rsrp_range = [-140, -44]
        rsrq_range = [-19.5, -3]
        offset_range = [-15, 15]

        RSRSP_TH: float = thresholds.get("rsrp", rsrp_range[0])
        RSRQ_TH: float = thresholds.get("rsrq", rsrq_range[0])
        OFFSET: float = thresholds.get("offset", offset_range[1])

        assert (
            rsrp_range[0] <= RSRSP_TH <= rsrp_range[1]
        ), f"rsrp threshold {RSRSP_TH} is out of range {rsrp_range}"
        assert (
            rsrq_range[0] <= RSRQ_TH <= rsrq_range[1]
        ), f"rsrq threshold {RSRQ_TH} is out of range {rsrq_range}"
        assert (
            offset_range[0] <= OFFSET <= offset_range[1]
        ), f"offset {OFFSET} is out of range {offset_range}"

        events_df = []
        size = self._measurement_points.shape[0]
        for i, row in self._measurement_points.iterrows():
            cells = row["lte"]
            if cells and 0 < i < size:
                # present cells - ta should be close to 0, ex. 0-10, mcc and mnc should exist
                scell = None
                for scell_index, cell_info in enumerate(cells):
                    if cell_info["identity"]["mMnc"] is not None:
                        scell = cell_info
                        break

                # previous cells
                prev_scell = None
                prev_cells = self._measurement_points.iloc[i - 1]["lte"]
                for cell_info in prev_cells:
                    if cell_info["identity"]["mMnc"] is not None:
                        prev_scell = cell_info
                        break

                assert scell is not None
                assert prev_scell is not None

                scell_rsrp = scell["signal"]["rsrp"]
                scell_rsrq = scell["signal"]["rsrq"]
                pscell_rsrp = prev_scell["signal"]["rsrp"]
                pscell_rsrq = prev_scell["signal"]["rsrq"]

                def is_a1_event():
                    return ((scell_rsrp > RSRSP_TH) and (pscell_rsrp < RSRSP_TH)) or (
                        (scell_rsrq > RSRQ_TH) and (pscell_rsrq < RSRQ_TH)
                    )

                def is_a2_event():
                    return ((scell_rsrp < RSRSP_TH) and (pscell_rsrp > RSRSP_TH)) or (
                        (scell_rsrq < RSRQ_TH) and (pscell_rsrq > RSRQ_TH)
                    )

                events_detected: list[str] = []
                if is_a1_event():
                    events_detected.append("A1")

                if is_a2_event():
                    events_detected.append("A2")

                for j, ncell in enumerate(cells):
                    if j == scell_index:  # shouldn't check serving cell against itself
                        continue

                    ncell_rsrp = ncell["signal"]["rsrp"]
                    ncell_rsrq = ncell["signal"]["rsrq"]

                    def is_a3_event():
                        return (ncell_rsrp >= (scell_rsrp + OFFSET)) or (
                            ncell_rsrq >= (scell_rsrq + OFFSET)
                        )

                    def is_a4_event():
                        return (ncell_rsrp >= RSRSP_TH) or (ncell_rsrq >= RSRQ_TH)

                    def is_a5_event():
                        return (
                            (scell_rsrp < RSRSP_TH) and (ncell_rsrp > RSRSP_TH)
                        ) or ((scell_rsrq < RSRQ_TH) and (ncell_rsrq > RSRQ_TH))

                    if is_a3_event():
                        events_detected.append("A3")

                    if is_a4_event():
                        events_detected.append("A4")

                    if is_a5_event():
                        events_detected.append("A5")

                for ev in set(events_detected):
                    events_df.append(
                        [
                            row["longitude"],
                            row["latitude"],
                            ev,
                            row["lte"][scell_index]["signal"].get(
                                "rsrp",
                                row["lte"][scell_index]["identity"].get("rsrp", None),
                            ),
                            row["lte"][scell_index]["signal"].get(
                                "rsrq",
                                row["lte"][scell_index]["identity"].get("rsrq", None),
                            ),
                            row["lte"][scell_index]["identity"].get(
                                "pci",
                                row["lte"][scell_index]["identity"].get("mPci", None),
                            ),
                            row["lte"][scell_index]["identity"].get("mMnc", None),
                            row["lte"][scell_index]["signal"].get(
                                "ta",
                                row["lte"][scell_index]["identity"].get("ta", None),
                            ),
                            row["lte"][scell_index]["signal"].get(
                                "rssi",
                                row["lte"][scell_index]["identity"].get("rssi", None),
                            ),
                            row["lte"][scell_index]["signal"].get(
                                "rssnr",
                                row["lte"][scell_index]["identity"].get("rssnr", None),
                            ),
                            row["lte"][scell_index]["signal"].get(
                                "cqiTableIndex", None
                            ),
                            row["lte"][scell_index]["signal"].get("cqi", None),
                            row["lte"][scell_index]["signal"].get("level", None),
                            row["lte"][scell_index]["identity"].get(
                                "earfcn",
                                row["lte"][scell_index]["identity"].get(
                                    "mEarfcn", None
                                ),
                            ),
                            row["lte"][scell_index]["identity"].get(
                                "tac",
                                row["lte"][scell_index]["identity"].get("mTac", None),
                            ),
                            row["lte"][scell_index]["identity"].get(
                                "bw",
                                row["lte"][scell_index]["identity"].get(
                                    "mBandwidth", None
                                ),
                            ),
                            row["lte"][scell_index]["identity"].get("mMcc", None),
                            row["lte"][scell_index]["identity"].get("dbm", None),
                        ]
                    )

        self.event_points = pd.DataFrame(
            events_df,
            columns=[
                "longitude",
                "latitude",
                "event",
                "rsrp",
                "rsrq",
                "pci",
                "mnc",
                "ta",
                "rssi",
                "rssnr",
                "cqi_table_index",
                "cqi",
                "level",
                "earfcn",
                "tac",
                "bandwidth",
                "mcc",
                "dbm",
            ],
        )
        return self.event_points

    def generate_grid(self, padding: float, chunks_count: int):
        min_longitude, max_longitude = (
            self.event_points["longitude"].min() - padding,
            self.event_points["longitude"].max() + padding,
        )
        min_latitude, max_latitude = (
            self.event_points["latitude"].min() - padding,
            self.event_points["latitude"].max() + padding,
        )

        def chunk_pattern(min_long, min_lat, max_long, max_lat):
            return [
                [min_long, min_lat],
                [max_long, min_lat],
                [max_long, max_lat],
                [min_long, max_lat],
                [min_long, min_lat],
            ]

        width: float = max_longitude - min_longitude
        height: float = max_latitude - min_latitude

        chunk_width = width / chunks_count
        chunk_height = height / chunks_count

        gridlines: list = []  # storage for all them chonks

        # columns
        for i in range(chunks_count):
            min_long = min_longitude + i * chunk_width
            max_long = min_longitude + (i + 1) * chunk_width
            chunks = [chunk_pattern(min_long, min_latitude, max_long, max_latitude)]
            gridlines.extend(chunks)

        # rows
        for i in range(chunks_count):
            min_lat = min_latitude + i * chunk_height
            max_lat = min_latitude + (i + 1) * chunk_height
            chunks = [chunk_pattern(min_longitude, min_lat, max_longitude, max_lat)]
            gridlines.extend(chunks)

        self.grid = gridlines

        def add_columns_to_df(df: pd.DataFrame) -> None:
            def coordinates_to_chunk_label(
                longitude: float, latitude: float
            ) -> tuple[int, int]:
                label_x = int((longitude - min_longitude) // chunk_width)
                label_y = int((latitude - min_latitude) // chunk_height)
                return label_x, label_y

            df["chunk"] = df.apply(
                lambda point: coordinates_to_chunk_label(
                    point["longitude"], point["latitude"]
                ),
                axis=1,
            )

            df["label"] = df.apply(
                lambda point: point["chunk"][0] * chunks_count + point["chunk"][1],
                axis=1,
            )

        # for raw data, some points may actually lie outside of the grid
        add_columns_to_df(self.event_points)
        add_columns_to_df(self._measurement_points)

        return self.grid

    def _prune_invalid_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        log.info("Pruning invalid data...")
        pruned_data = raw_data[raw_data["nw data"].apply(lambda x: len(str(x)) > 1)]
        return pruned_data

    def _parse_to_measurement_points(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        log.info("Parsing data to measurement points...")
        measurements = explode_nw_data(raw_data)
        measurements = expand_dataframe(measurements)

        # remove rows with cells which are not lte
        measurements = measurements[
            measurements["plmn lte"].apply(lambda x: len(x) > 0)
        ]

        # extract lte data from 'nw data' and convert str to correct types
        measurements["lte"] = measurements["nw data"].apply(lambda x: x.lte())
        measurements["lte"] = measurements["lte"].apply(self._to_dict)
        measurements["lte"] = measurements["lte"].apply(self._convert_types)
        measurements["time"] = measurements["time"].apply(pd.to_datetime)
        measurements[["longitude", "latitude"]] = measurements[
            ["longitude", "latitude"]
        ].apply(pd.to_numeric)

        # remove unnecessary columns
        measurements.drop("nw data", axis=1, inplace=True)
        measurements.drop("plmn wcdma", axis=1, inplace=True)

        # rename 'plmn lte' to 'mcc_mnc'
        measurements.rename(columns={"plmn lte": "mcc-mnc"}, inplace=True)
        return measurements

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
            "miuiLevel",
            "mOptimizedLevel",
            "parametersUseForLevel",
        ]
        identity_columns = [
            "mCi",
            "mPci",
            "mTac",
            "mEarfcn",
            "mBandwidth",
            "mMcc",
            "mMnc",
            # other convention
            "pci",
            "tac",
            "earfcn",
            "bw",
            "rsrp",
            "rsrq",
            "rssi",
            "dbm",
            "ta",
        ]
        for info in cell_infos:
            for column in signal_columns:
                try:
                    info["signal"][column] = pd.to_numeric(
                        info["signal"][column], errors="coerce", downcast="integer"
                    )
                    # if info["signal"][column] > 2147483640:
                    #     info["signal"][column] = float("inf")
                except KeyError:
                    pass
                except:
                    pass

            for column in identity_columns:
                try:
                    info["identity"][column] = int(info["identity"][column])
                except KeyError:
                    pass
                except ValueError:
                    info["identity"][column] = None
                except:
                    pass

        return cell_infos
