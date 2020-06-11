import pandas as pd
import numpy as np


class TimeCovariates(object):
    def __init__(self, start_date, num_ts=100, freq="H", normalized=True):
        self.start_date = start_date
        self.num_ts = num_ts
        self.freq = freq
        self.normalized = normalized
        self.dti = pd.date_range(self.start_date, periods=self.num_ts, freq=self.freq)

    def _minute_of_hour(self):
        minutes = np.array(self.dti.minute, dtype=np.float)
        if self.normalized:
            minutes = minutes / 59.0 - 0.5
        return minutes

    def _hour_of_day(self):
        hours = np.array(self.dti.hour, dtype=np.float)
        if self.normalized:
            hours = hours / 23.0 - 0.5
        return hours

    def _day_of_week(self):
        dayWeek = np.array(self.dti.dayofweek, dtype=np.float)
        if self.normalized:
            dayWeek = dayWeek / 6.0 - 0.5
        return dayWeek

    def _day_of_month(self):
        dayMonth = np.array(self.dti.day, dtype=np.float)
        if self.normalized:
            dayMonth = dayMonth / 30.0 - 0.5
        return dayMonth

    def _day_of_year(self):
        dayYear = np.array(self.dti.dayofyear, dtype=np.float)
        if self.normalized:
            dayYear = dayYear / 364.0 - 0.5
        return dayYear

    def _month_of_year(self):
        monthYear = np.array(self.dti.month, dtype=np.float)
        if self.normalized:
            monthYear = monthYear / 11.0 - 0.5
        return monthYear

    def _week_of_year(self):
        weekYear = np.array(self.dti.weekofyear, dtype=np.float)
        if self.normalized:
            weekYear = weekYear / 51.0 - 0.5
        return weekYear

    def get_covariates(self):
        MOH = self._minute_of_hour().reshape(1, -1)
        HOD = self._hour_of_day().reshape(1, -1)
        DOM = self._day_of_month().reshape(1, -1)
        DOW = self._day_of_week().reshape(1, -1)
        DOY = self._day_of_year().reshape(1, -1)
        MOY = self._month_of_year().reshape(1, -1)
        WOY = self._week_of_year().reshape(1, -1)

        all_covs = [MOH, HOD, DOM, DOW, DOY, MOY, WOY]

        return np.vstack(all_covs)
