import numpy as np

class FrequencyProcessor:

    @staticmethod
    def fft(t, x, wmin, wmax, c=10):
        dt = np.mean(t[1:]-t[:-1])
        freqs = np.fft.fftfreq(c * len(t), dt)
        y = np.fft.fft(x, n=c * len(t))
        y = np.abs(y)
        y = y[(freqs >= wmin) & (freqs <= wmax)]
        freqs = freqs[(freqs >= wmin) & (freqs <= wmax)]
        return freqs, y

    @staticmethod
    def keyw(w, fit, crit=0.9):
        kfit = []
        kw = []
        mxidx = np.argmax(np.abs(fit))
        mx = abs(fit[mxidx])
        for idx in range(len(fit)):
            if abs(fit[idx]) >= crit * mx:
                kw.append(w[idx])
                kfit.append(fit[idx])
        return kw, kfit

    @staticmethod
    def keywsort(w, fit, crit=0.9, crit_count=4):
        kfit = []
        kw = []
        idxs = np.argsort(fit)[::-1]
        sortfit = np.array([fit[i] for i in idxs])
        sortw = np.array([w[i] for i in idxs])
        mx = sortfit[0]
        count = 0
        for i in range(1, len(sortfit)):
            if sortfit[i] > crit * mx or count < crit_count:
                kfit.append(sortfit[i])
                kw.append(sortw[i])
                count += 1
            else:
                break
        return kw, kfit

    @staticmethod
    def findmaxfreqs(w, fit):
        kfit = []
        kw = []
        for idx in range(1, len(w) - 1):
            if fit[idx] > fit[idx - 1] and fit[idx] > fit[idx + 1]:
                kw.append(w[idx])
                kfit.append(fit[idx])
        return kw, kfit

    @staticmethod
    def findmaxfreq(w, fit):
        idxs = np.argsort(fit)[::-1]
        return w[idxs[0]]

    @staticmethod
    def find_dif_in_freqs(w, w0):
        for i in range(len(w)):
            for j in range(i + 1, len(w)):
                for freq0 in w0:
                    if abs((0.5 * (w[i] + w[j]) - freq0) / (0.5 * (0.5 * (w[i] + w[j]) + freq0))) < 0.05:
                        return abs(w[i] - w[j]) / 2
        return None

    @staticmethod
    def find_freq_for_summand(t, x, wmin, wmax, c=10):
        freqs, y = FrequencyProcessor.fft(t, x, wmin, wmax, c)
        y = np.abs(y)
        return FrequencyProcessor.findmaxfreqs(freqs, y)

    @staticmethod
    def find_freq_for_multiplier(t, x, wmin, wmax, w0, c=10, crit=0.9, crit_count=4):
        freqs, y = FrequencyProcessor.fft(t, x, wmin, wmax, c)
        y = np.abs(y)
        freqs, y = FrequencyProcessor.findmaxfreqs(freqs, y)
        freqs, y = FrequencyProcessor.keywsort(freqs, y, crit, crit_count)
        return FrequencyProcessor.find_dif_in_freqs(freqs, w0)
