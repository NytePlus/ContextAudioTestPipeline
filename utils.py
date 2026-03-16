def hotword_process(hotword_string):
    hotwords = hotword_string.split(',')
    hotwords = [hw.strip().lower().capitalize() for hw in hotwords]
    return ', '.join(hotwords)