from timeit import default_timer as moment


class PrettyPrintTime:
    def __init__(self):
        self.start = 0
        self.checkpoint = [0]

    def restart(self):
        self.start = moment()
        self.checkpoint = [self.start]

    def time_format(self, sec):
        secs = round(sec % 60)
        mins = round((sec // 60) % 60)
        hrs = round(sec // 60 // 60)

        stamp = "{:02d}:{:02d}:{:02d}".format(hrs, mins, secs)

        return stamp

    def the_time(self):
        self.start = self.checkpoint[-1]
        self.checkpoint.append(moment())
        print(self.time_format(self.checkpoint[-1] - self.start))

    def the_total_time(self):
        print(self.time_format(moment() - self.checkpoint[0]))
