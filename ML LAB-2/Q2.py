#Q2
economicstatus = output.apply(lambda x: "rich" if x > 200 else "poor")
print(economicstatus)
