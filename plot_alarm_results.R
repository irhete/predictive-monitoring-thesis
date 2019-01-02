library(ggplot2)
library(plyr)
library(RColorBrewer)
library(reshape2)
library(scales)

setwd("~/Repos/predictive-monitoring-thesis/results_alarms")
files <- list.files()
data <- data.frame()
for (file in files) {
  tmp <- read.table(file, sep=";", header=T)
  data <- rbind(data, tmp)  
}
head(data, 10)

data$metric <- gsub("_mean", "", data$metric)
data$ratio <- paste(data$c_miss, data$c_action, sep=":")

base_size = 28
line_size = 0.8
point_size = 3
cross_size = 4.5

data$alarm_method <- as.character(data$alarm_method)
data$dataset <- as.character(data$dataset)
data$alarm_method[data$alarm_method=="fixed0"] <- "always alarm"
data$alarm_method[data$alarm_method=="fixed110"] <- "never alarm"
data$alarm_method[data$alarm_method=="fixed50"] <- "tau=0.5"
data$alarm_method[data$alarm_method=="opt_threshold"] <- "optimized"
data[data$dataset == "bpic2011_f1","dataset"] <- "bpic2011_1"
data[data$dataset == "bpic2011_f2","dataset"] <- "bpic2011_2"
data[data$dataset == "bpic2011_f3","dataset"] <- "bpic2011_3"
data[data$dataset == "bpic2011_f4","dataset"] <- "bpic2011_4"

data[data$dataset == "bpic2015_1_f2","dataset"] <- "bpic2015_1"
data[data$dataset == "bpic2015_2_f2","dataset"] <- "bpic2015_2"
data[data$dataset == "bpic2015_3_f2","dataset"] <- "bpic2015_3"
data[data$dataset == "bpic2015_4_f2","dataset"] <- "bpic2015_4"
data[data$dataset == "bpic2015_5_f2","dataset"] <- "bpic2015_5"

data[data$dataset == "insurance_activity","dataset"] <- "insurance_1"
data[data$dataset == "insurance_followup","dataset"] <- "insurance_2"

data[data$dataset == "sepsis_cases_1","dataset"] <- "sepsis_1"
data[data$dataset == "sepsis_cases_2","dataset"] <- "sepsis_2"
data[data$dataset == "sepsis_cases_4","dataset"] <- "sepsis_3"

data[data$dataset == "bpic2012_accepted","dataset"] <- "bpic2012_1"
data[data$dataset == "bpic2012_declined","dataset"] <- "bpic2012_2"
data[data$dataset == "bpic2012_cancelled","dataset"] <- "bpic2012_3"

data[data$dataset == "bpic2017_accepted","dataset"] <- "bpic2017_1"
data[data$dataset == "bpic2017_refused","dataset"] <- "bpic2017_2"
data[data$dataset == "bpic2017_cancelled","dataset"] <- "bpic2017_3"

data[data$dataset == "traffic_fines_1","dataset"] <- "traffic"
data[data$dataset == "hospital_billing_2","dataset"] <- "hospital_1"
data[data$dataset == "hospital_billing_3","dataset"] <- "hospital_2"

data[data$dataset == "dc","dataset"] <- "DR"
data[data$dataset == "crm2","dataset"] <- "LtC"


head(data)
data[is.na(data$value), "value"] <- 0

color_palette <- c("#0072B2", "#000000", "#E69F00", "#009E73", "#56B4E9","#D55E00", "#999999", "#F0E442", "#CC79A7")
png("~/Dropbox/phd_thesis_irene/alarm_paper/results_ratios.png", width=1000, height=1750)
ggplot(subset(data, c_postpone==0 & metric=="cost_avg" & !grepl("fixed", alarm_method)), aes(x=c_miss, y=value, color=alarm_method, shape=alarm_method)) + 
  geom_point(size=point_size) + geom_line(size=line_size) + scale_x_continuous(breaks=c(3,10,20),
                                                                               labels=c("3:1", "10:1", "20:1"))+
  theme_bw(base_size=base_size) + ylab("Avg. cost per case") + xlab("c_out : c_in") + facet_wrap( ~ dataset, ncol=4) +
  scale_color_manual(values=color_palette) + theme(legend.position="top")
dev.off()

png("~/Dropbox/phd_thesis_irene/alarm_paper/results_ratios_selected.png", width=1000/4*3, height=1750/7*2)
ggplot(subset(data, c_postpone==0 & metric=="cost_avg" & !grepl("fixed", alarm_method) & 
                dataset %in% c("bpic2017_2", "bpic2011_1", "production", "bpic2015_1")), aes(x=c_miss, y=value, color=alarm_method, shape=alarm_method)) + 
  geom_point(size=point_size) + geom_line(size=line_size) + scale_x_continuous(breaks=c(3,10,20),
                                                                               labels=c("3:1", "10:1", "20:1"))+
  theme_bw(base_size=base_size) + ylab("Avg. cost per case") + xlab("c_out : c_in") + facet_wrap( ~ dataset, ncol=2) +
  scale_color_manual(values=color_palette) + theme(legend.position="right")
dev.off()

color_palette <- c("#0072B2", "#000000", "#E69F00", "#009E73", "#56B4E9","#D55E00", "#999999", "#F0E442", "#CC79A7")
tmp <- subset(data, c_postpone==0 & metric=="cost_avg" & (c_miss %in% c(1,2,5,20)))
tmp$ratio <- factor(tmp$ratio, levels(factor(tmp$ratio))[c(1,3,4,2)])
png("~/Dropbox/phd_thesis_irene/alarm_paper/results_thresholds.png", width=1000, height=1750)
ggplot(tmp, aes(x=threshold, y=value, group=ratio, color=ratio, shape=ratio)) + 
  geom_point(size=point_size) + geom_line(size=line_size) + geom_point(data=subset(tmp, alarm_method=="optimized"), size=cross_size, color="red", stroke=1.5, shape=4, aes(x=threshold, y=value, color=factor(c_miss))) +
  theme_bw(base_size=base_size) + ylab("Avg. cost per case") + xlab(expression("Threshold ("~tau~" )")) + 
  facet_wrap( ~ dataset, ncol=4) + scale_color_manual(values=color_palette, name="c_out : c_in") + theme(legend.position="top") +
  scale_shape(guide=FALSE)
dev.off()

png("~/Dropbox/phd_thesis_irene/alarm_paper/results_ratios_fscore_earliness.png", width=1000, height=1750)
ggplot(subset(data, c_postpone==0 & metric %in% c("fscore", "earliness")  & !grepl("fixed", alarm_method)), aes(x=c_miss, y=value, color=metric, shape=metric)) + 
  geom_point(size=point_size) + geom_line(size=line_size) + scale_x_continuous(breaks=c(3,10,20),
                                                                               labels=c("3:1", "10:1", "20:1"))+
  theme_bw(base_size=base_size) + ylab("Avg. cost per case") + xlab("c_out : c_in") + facet_wrap( ~ dataset, ncol=4) +
  scale_color_manual(values=color_palette[c(1,9)]) + theme(legend.position="top")
dev.off()

### heatmaps effectiveness

setwd("~/Repos/predictive-monitoring-thesis/results_alarms_eff")
files <- list.files()
data <- data.frame()
for (file in files) {
  tmp <- read.table(file, sep=";", header=T)
  data <- rbind(data, tmp)  
}

data$metric <- gsub("_mean", "", data$metric)
data$ratio <- paste(data$c_miss, data$c_action, sep=":")
data$early_type <- as.character(data$early_type)
data$dataset <- as.character(data$dataset)
data[data$dataset == "bpic2011_f1","dataset"] <- "bpic2011_1"
data[data$dataset == "bpic2011_f2","dataset"] <- "bpic2011_2"
data[data$dataset == "bpic2011_f3","dataset"] <- "bpic2011_3"
data[data$dataset == "bpic2011_f4","dataset"] <- "bpic2011_4"

data[data$dataset == "bpic2015_1_f2","dataset"] <- "bpic2015_1"
data[data$dataset == "bpic2015_2_f2","dataset"] <- "bpic2015_2"
data[data$dataset == "bpic2015_3_f2","dataset"] <- "bpic2015_3"
data[data$dataset == "bpic2015_4_f2","dataset"] <- "bpic2015_4"
data[data$dataset == "bpic2015_5_f2","dataset"] <- "bpic2015_5"

data[data$dataset == "insurance_activity","dataset"] <- "insurance_1"
data[data$dataset == "insurance_followup","dataset"] <- "insurance_2"

data[data$dataset == "sepsis_cases_1","dataset"] <- "sepsis_1"
data[data$dataset == "sepsis_cases_2","dataset"] <- "sepsis_2"
data[data$dataset == "sepsis_cases_4","dataset"] <- "sepsis_3"

data[data$dataset == "bpic2012_accepted","dataset"] <- "bpic2012_1"
data[data$dataset == "bpic2012_declined","dataset"] <- "bpic2012_2"
data[data$dataset == "bpic2012_cancelled","dataset"] <- "bpic2012_3"

data[data$dataset == "bpic2017_accepted","dataset"] <- "bpic2017_1"
data[data$dataset == "bpic2017_refused","dataset"] <- "bpic2017_2"
data[data$dataset == "bpic2017_cancelled","dataset"] <- "bpic2017_3"

data[data$dataset == "traffic_fines_1","dataset"] <- "traffic"
data[data$dataset == "hospital_billing_2","dataset"] <- "hospital_1"
data[data$dataset == "hospital_billing_3","dataset"] <- "hospital_2"

data[data$dataset == "dc","dataset"] <- "DR"
data[data$dataset == "crm2","dataset"] <- "LtC"

head(data)
names(data)[names(data)=="c_com"] <- "eff"

dt_as_is <- subset(data, metric=="cost_avg_baseline")
dt_to_be <- subset(data, metric=="cost_avg")
dt_merged <- merge(dt_as_is, dt_to_be, by=c("dataset", "method", "alarm_method", "c_miss", "c_action", "c_postpone", "eff", "early_type", "cls", "ratio"), suffixes=c("_as_is", "_to_be"))

dt_merged$benefit <- dt_merged$value_as_is - dt_merged$value_to_be
dt_merged$ratio <- as.factor(dt_merged$ratio)
dt_merged$ratio <- factor(dt_merged$ratio, levels(dt_merged$ratio)[c(2,4,5,7,1,3,6)])

color_low <- "#ef8a62"
color_high <- "#67a9cf"

#color_low <- "#af8dc3"
#color_high <- "#7fbf7b"

png("~/Dropbox/phd_thesis_irene/alarm_paper/results_effectiveness_const.png", width=1000, height=1850)
ggplot(subset(dt_merged, c_miss %in% c(1,2,3,5,10,20) & grepl("const", early_type)), aes(eff, factor(ratio))) + 
  geom_tile(aes(fill = benefit), colour = "white") + scale_x_continuous(breaks=c(0,0.2,0.4,0.6,0.8,1), labels=c("0",".2",".4",".6",".8","1"))+
  theme_bw(base_size=base_size) + scale_fill_gradient2(low = muted(color_low), high = muted(color_high)) + facet_wrap( ~ dataset, ncol=4) + 
  xlab("mitigation effectiveness (eff)") + ylab("c_out : c_in") + theme(legend.position="top", legend.text = element_text(size=24), legend.key.size = unit(2, "cm"))
dev.off()

png("~/Dropbox/phd_thesis_irene/alarm_paper/results_effectiveness_linear.png", width=1000, height=1850)
ggplot(subset(dt_merged, c_miss %in% c(1,2,3,5,10,20) & grepl("linear", early_type)), aes(eff, factor(ratio))) + 
  geom_tile(aes(fill = benefit), colour = "white") + scale_x_continuous(breaks=c(0,0.2,0.4,0.6,0.8,1), labels=c("0",".2",".4",".6",".8","1"))+
  theme_bw(base_size=base_size) + scale_fill_gradient2(low = muted(color_low), high = muted(color_high)) + facet_wrap( ~ dataset, ncol=4) + 
  xlab("mitigation effectiveness (eff)") + ylab("c_out : c_in") + theme(legend.position="top", legend.text = element_text(size=24), legend.key.size = unit(2, "cm"))
dev.off()


### heatmaps cost of compensation
setwd("~/Repos/predictive-monitoring-thesis/results_alarms_ccom")
files <- list.files()
data <- data.frame()
for (file in files) {
  tmp <- read.table(file, sep=";", header=T)
  data <- rbind(data, tmp)  
}

data$metric <- gsub("_mean", "", data$metric)
data$ratio <- paste(data$c_miss, data$c_action, sep=":")
data$ratio_com <- ifelse(data$c_com==0, "1:0", ifelse(data$c_com > 1, sprintf("1:%s", data$c_com), sprintf("%s:1", 1/data$c_com)))
data$dataset <- as.character(data$dataset)
data$early_type <- as.character(data$early_type)
data[data$dataset == "bpic2011_f1","dataset"] <- "bpic2011_1"
data[data$dataset == "bpic2011_f2","dataset"] <- "bpic2011_2"
data[data$dataset == "bpic2011_f3","dataset"] <- "bpic2011_3"
data[data$dataset == "bpic2011_f4","dataset"] <- "bpic2011_4"

data[data$dataset == "bpic2015_1_f2","dataset"] <- "bpic2015_1"
data[data$dataset == "bpic2015_2_f2","dataset"] <- "bpic2015_2"
data[data$dataset == "bpic2015_3_f2","dataset"] <- "bpic2015_3"
data[data$dataset == "bpic2015_4_f2","dataset"] <- "bpic2015_4"
data[data$dataset == "bpic2015_5_f2","dataset"] <- "bpic2015_5"

data[data$dataset == "insurance_activity","dataset"] <- "insurance_1"
data[data$dataset == "insurance_followup","dataset"] <- "insurance_2"

data[data$dataset == "sepsis_cases_1","dataset"] <- "sepsis_1"
data[data$dataset == "sepsis_cases_2","dataset"] <- "sepsis_2"
data[data$dataset == "sepsis_cases_4","dataset"] <- "sepsis_3"

data[data$dataset == "bpic2012_accepted","dataset"] <- "bpic2012_1"
data[data$dataset == "bpic2012_declined","dataset"] <- "bpic2012_2"
data[data$dataset == "bpic2012_cancelled","dataset"] <- "bpic2012_3"

data[data$dataset == "bpic2017_accepted","dataset"] <- "bpic2017_1"
data[data$dataset == "bpic2017_refused","dataset"] <- "bpic2017_2"
data[data$dataset == "bpic2017_cancelled","dataset"] <- "bpic2017_3"

data[data$dataset == "traffic_fines_1","dataset"] <- "traffic"
data[data$dataset == "hospital_billing_2","dataset"] <- "hospital_1"
data[data$dataset == "hospital_billing_3","dataset"] <- "hospital_2"

data[data$dataset == "dc","dataset"] <- "DR"
data[data$dataset == "crm2","dataset"] <- "LtC"

head(data)

dt_as_is <- subset(data, metric=="cost_avg_baseline")
dt_to_be <- subset(data, metric=="cost_avg")
dt_merged <- merge(dt_as_is, dt_to_be, by=c("dataset", "method", "alarm_method", "c_miss", "c_action", "c_postpone", "c_com", "early_type", "cls", "ratio", "ratio_com"), suffixes=c("_as_is", "_to_be"))

dt_merged$benefit <- dt_merged$value_as_is - dt_merged$value_to_be
dt_merged$ratio <- as.factor(dt_merged$ratio)
dt_merged$ratio <- factor(dt_merged$ratio, levels(dt_merged$ratio)[c(2,4,5,7,1,3,6)])
dt_merged$ratio_com <- as.factor(dt_merged$ratio_com)
dt_merged$ratio_com <- factor(dt_merged$ratio_com, levels(dt_merged$ratio_com)[c(1,13,10,2,14,12,11,3,5,7,9,4,6,8)])

text_size <- 12

png("~/Dropbox/phd_thesis_irene/alarm_paper/results_compensation_const.png", width=1000, height=1850)
ggplot(subset(dt_merged, c_miss %in% c(1,2,3,5,10,20) & grepl("const", early_type) & ratio!="3:1" & 
                !(ratio_com%in%c("3:1", "1:3", "1:40", "40:1", NA, "20:1", "1:20"))), aes(ratio_com, factor(ratio))) + geom_tile(aes(fill = benefit), colour = "white") + 
  theme_bw(base_size=base_size) + scale_fill_gradient2(low = muted(color_low), high = muted(color_high)) + facet_wrap(.~dataset, ncol=4) + 
  #scale_x_discrete(labels=c(expression(frac(1, 0)), expression(frac(10, 1)), expression(frac(5, 1)), expression(frac(2, 1)), expression(frac(1, 1)), expression(frac(1, 2)), expression(frac(1, 5)), expression(frac(1, 10)))) +
  #scale_x_discrete(labels=c("1:0", "\n10:1", "5:1", "\n2:1", "1:1", "\n1:2", "1:5", "\n1:10")) +
  xlab("c_in : c_com") + ylab("c_out : c_in") + theme(legend.position="top", legend.text = element_text(size=24), legend.key.size = unit(2, "cm"),
                                                      axis.text.x = element_text(angle = 60, hjust = 1))
dev.off()

png("~/Dropbox/phd_thesis_irene/alarm_paper/results_compensation_linear.png", width=1000, height=1850)
ggplot(subset(dt_merged, c_miss %in% c(1,2,3,5,10,20) & grepl("linear", early_type) & ratio!="3:1" & 
                !(ratio_com%in%c("3:1", "1:3", "1:40", "40:1", NA, "20:1", "1:20"))), aes(ratio_com, factor(ratio))) + geom_tile(aes(fill = benefit), colour = "white") + 
  theme_bw(base_size=base_size) + scale_fill_gradient2(low = muted(color_low), high = muted(color_high)) + facet_wrap(.~dataset, ncol=4) + 
  xlab("c_in : c_com") + ylab("c_out : c_in") +  theme(legend.position="top", legend.text = element_text(size=24), legend.key.size = unit(2, "cm"),
                                                       axis.text.x = element_text(angle = 60, hjust = 1))
dev.off()

