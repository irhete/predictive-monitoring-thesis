library(ggplot2)

setwd("~/Repos/predictive-monitoring-thesis/results_stability/")
files <- list.files()
files

data <- data.frame()
for (file in files) {
  tmp <- read.table(file, sep=";", header=TRUE)
  data <- rbind(data, tmp)
}

head(subset(data, method=="prefix_index"), 43)

data <- subset(data, grepl("calibrated", cls))
data[data$cls=="lstm_calibrated", "Method"] <- "LSTM"
data[data$cls=="rf_calibrated" & data$method=="single_index", "Method"] <- "RF_idx_pad"
data[data$cls=="rf_calibrated" & data$method=="prefix_index", "Method"] <- "RF_idx_mul"
data[data$cls=="rf_calibrated" & data$method=="single_agg", "Method"] <- "RF_agg"
data[data$cls=="xgboost_calibrated" & data$method=="single_index", "Method"] <- "XGB_idx_pad"
data[data$cls=="xgboost_calibrated" & data$method=="prefix_index", "Method"] <- "XGB_idx_mul"
data[data$cls=="xgboost_calibrated" & data$method=="single_agg", "Method"] <- "XGB_agg"

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

datasets1 <- c("bpic2011_1", "bpic2011_2", "bpic2011_3", "bpic2011_4", "bpic2015_1", "bpic2015_2", "bpic2015_3", 
               "bpic2015_4", "bpic2015_5", "production", "insurance_1", "insurance_2")
datasets2 <- c("sepsis_1", "sepsis_2", "sepsis_3", "bpic2012_1", "bpic2012_2", "bpic2012_3", "bpic2017_1", 
               "bpic2017_2", "bpic2017_3", "traffic", "hospital_1", "hospital_2", "DR", "github", "LtC")

base_size = 32
line_size = 0.8
point_size = 3.5
width = 1300
height = 1200

data <- subset(data, !is.na(data$score))

color_palette <- c("#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")[c(1,8,4,6,2,7,3)]
png("/home/irene/Dropbox/phd_thesis_irene/stability_paper/aucs_calibrated_3cols_1.png", width=width, height=height)
ggplot(subset(data, metric=="auc" & dataset %in% datasets1), aes(x=as.numeric(nr_events), y=score, color=Method, shape=Method)) + 
  geom_point(size=point_size) + geom_line(size=line_size) + 
  theme_bw(base_size=base_size) + facet_wrap(~dataset, scales="free", ncol=3) + scale_color_manual(values=color_palette) + 
  labs(x="Prefix length", y="AUC") + theme(legend.position = 'top', legend.key.size = unit(1.5, 'lines'))
dev.off()

png("/home/irene/Dropbox/phd_thesis_irene/stability_paper/aucs_calibrated_3cols_2.png", width=width, height=height*5/4)
ggplot(subset(data, metric=="auc" & dataset %in% datasets2), aes(x=as.numeric(nr_events), y=score, color=Method, shape=Method)) + 
  geom_point(size=point_size) + geom_line(size=line_size) + 
  theme_bw(base_size=base_size) + facet_wrap(~dataset, scales="free", ncol=3) + scale_color_manual(values=color_palette) + 
  labs(x="Prefix length", y="AUC") + theme(legend.position = 'top', legend.key.size = unit(1.5, 'lines'))
dev.off()

text_size <- 4
color_palette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")[c(1,8,4,6,2,7,3)]
png("/home/irene/Dropbox/phd_thesis_irene/stability_paper/stability_mean_abs_calibrated_3cols_1.png", width=width, height=height)
ggplot(subset(data, metric=="stability" & dataset %in% datasets1), aes(x=Method, y=score, fill=Method)) + geom_bar(stat="identity", color="black") + 
  theme_bw(base_size=base_size) + facet_wrap(~dataset, scales="free", ncol=3) + geom_text(aes(label=round(score, 3)), size=5, vjust=-0.25) + 
  scale_fill_manual(values=color_palette) + theme(axis.text.x=element_blank(), legend.position = 'top',
                                                  legend.key.size = unit(1.5, 'lines')) + #ylim(c(0, 1.05)) +
  ylab("Temporal stability") + xlab("") +  coord_cartesian(ylim=c(0.65, 1.05))
dev.off()

png("/home/irene/Dropbox/phd_thesis_irene/stability_paper/stability_mean_abs_calibrated_3cols_2.png", width=width, height=height*5/4)
ggplot(subset(data, metric=="stability" & dataset %in% datasets2), aes(x=Method, y=score, fill=Method)) + geom_bar(stat="identity", color="black") + 
  theme_bw(base_size=base_size) + facet_wrap(~dataset, scales="free", ncol=3) + geom_text(aes(label=round(score, 3)), size=5, vjust=-0.25) + 
  scale_fill_manual(values=color_palette) + theme(axis.text.x=element_blank(), legend.position = 'top',
                                                  legend.key.size = unit(1.5, 'lines')) + #ylim(c(0, 1.05)) +
  ylab("Temporal stability") + xlab("") +  coord_cartesian(ylim=c(0.65, 1.05))
dev.off()



# smoothed
setwd("~/Repos/predictive-monitoring-thesis/results_stability_smoothed/")
files <- list.files()
files

data <- data.frame()
for (file in files) {
  tmp <- read.table(file, sep=";", header=TRUE)
  data <- rbind(data, tmp)
}

data[data$cls=="lstm_calibrated", "Method"] <- "LSTM"
data[data$cls=="rf_calibrated" & data$method=="single_index", "Method"] <- "RF_idx_pad"
data[data$cls=="rf_calibrated" & data$method=="prefix_index", "Method"] <- "RF_idx_mul"
data[data$cls=="rf_calibrated" & data$method=="single_agg", "Method"] <- "RF_agg"
data[data$cls=="xgboost_calibrated" & data$method=="single_index", "Method"] <- "XGB_idx_pad"
data[data$cls=="xgboost_calibrated" & data$method=="prefix_index", "Method"] <- "XGB_idx_mul"
data[data$cls=="xgboost_calibrated" & data$method=="single_agg", "Method"] <- "XGB_agg"

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

datasets1 <- c("bpic2011_1", "bpic2011_2", "bpic2011_3", "bpic2011_4", "bpic2015_1", "bpic2015_2", "bpic2015_3", 
               "bpic2015_4", "bpic2015_5", "production", "insurance_1", "insurance_2")
datasets2 <- c("sepsis_1", "sepsis_2", "sepsis_3", "bpic2012_1", "bpic2012_2", "bpic2012_3", "bpic2017_1", 
               "bpic2017_2", "bpic2017_3", "traffic", "hospital_1", "hospital_2", "DR", "github", "LtC")

head(data)

color_palette <- c("#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")[c(1,8,4,6,2,7,3)]
png("/home/irene/Dropbox/phd_thesis_irene/stability_paper/aucs_all_3cols_1.png", width=width, height=height)
ggplot(subset(data, metric=="auc" & dataset %in% datasets1), aes(x=beta, y=score, color=Method, shape=Method)) + geom_point(size=point_size) + geom_line(size=line_size) + 
  theme_bw(base_size=base_size) + 
  facet_wrap(~dataset, scales="free", ncol=3) + scale_color_manual(values=color_palette) + labs(x="Alpha", y="AUC") + theme(legend.position = 'top',
                                                                                                                            legend.key.size = unit(1.5, 'lines'))

dev.off()

png("/home/irene/Dropbox/phd_thesis_irene/stability_paper/aucs_all_3cols_2.png", width=width, height=height*5/4)
ggplot(subset(data, metric=="auc" & dataset %in% datasets2), aes(x=beta, y=score, color=Method, shape=Method)) + geom_point(size=point_size) + geom_line(size=line_size) + 
  theme_bw(base_size=base_size) + 
  facet_wrap(~dataset, scales="free", ncol=3) + scale_color_manual(values=color_palette) + labs(x="Alpha", y="AUC") + theme(legend.position = 'top',
                                                                                                                            legend.key.size = unit(1.5, 'lines'))

dev.off()

png("/home/irene/Dropbox/phd_thesis_irene/stability_paper/stability_all_3cols_1.png", width=width, height=height)
ggplot(subset(data, metric=="stability" & dataset %in% datasets1), aes(x=beta, y=score, color=Method, shape=Method)) + geom_point(size=point_size) + geom_line(size=line_size) + 
  theme_bw(base_size=base_size) + 
  facet_wrap(~dataset, scales="free", ncol=3) + scale_color_manual(values=color_palette) + labs(x="Alpha", y="AUC") + theme(legend.position = 'top',
                                                                                                                            legend.key.size = unit(1.5, 'lines'))

dev.off()

png("/home/irene/Dropbox/phd_thesis_irene/stability_paper/stability_all_3cols_2.png", width=width, height=height*5/4)
ggplot(subset(data, metric=="stability" & dataset %in% datasets2), aes(x=beta, y=score, color=Method, shape=Method)) + geom_point(size=point_size) + geom_line(size=line_size) + 
  theme_bw(base_size=base_size) + 
  facet_wrap(~dataset, scales="free", ncol=3) + scale_color_manual(values=color_palette) + labs(x="Alpha", y="AUC") + theme(legend.position = 'top',
                                                                                                                            legend.key.size = unit(1.5, 'lines'))

dev.off()

library(scales)

dt_merged <- merge(subset(data, metric=="auc"), subset(data, metric=="stability"), by=c("dataset", "method", "cls", "beta", "Method"))

png("/home/irene/Dropbox/phd_thesis_irene/stability_paper/stability_vs_auc_3cols_1.png", width=width, height=height)
ggplot(subset(dt_merged, dataset %in% datasets1), aes(x=score.y, y=score.x, color=Method, shape=Method)) + geom_point(size=point_size) + geom_line(size=line_size) + 
  theme_bw(base_size=base_size) + 
  facet_wrap(~dataset, scales="free", ncol=3) + scale_color_manual(values=color_palette) + labs(x="Temporal stability", y="AUC") + theme(legend.position = 'top',
                                                                                                                                         legend.key.size = unit(1.5, 'lines')) +
  scale_x_continuous(breaks = trans_breaks(identity, identity, n = 3))

dev.off()

png("/home/irene/Dropbox/phd_thesis_irene/stability_paper/stability_vs_auc_3cols_2.png", width=width, height=height*5/4)
ggplot(subset(dt_merged, dataset %in% datasets2), aes(x=score.y, y=score.x, color=Method, shape=Method)) + geom_point(size=point_size) + geom_line(size=line_size) + 
  theme_bw(base_size=base_size) + 
  facet_wrap(~dataset, scales="free", ncol=3) + scale_color_manual(values=color_palette) + labs(x="Temporal stability", y="AUC") + theme(legend.position = 'top',
                                                                                                                                         legend.key.size = unit(1.5, 'lines')) +
  scale_x_continuous(breaks = trans_breaks(identity, identity, n = 3))

dev.off()

png("/home/irene/Dropbox/phd_thesis_irene/stability_paper/stability_vs_auc_selected.png", width=width*2/3, height=height*2/3)
ggplot(subset(dt_merged, dataset %in% c("bpic2017_2", "bpic2012_3", "bpic2011_2", "github")), aes(x=score.y, y=score.x, color=Method, shape=Method)) + geom_point(size=point_size) + geom_line(size=line_size) + 
  theme_bw(base_size=base_size-4) + 
  facet_wrap(~dataset, scales="free", ncol=2) + scale_color_manual(values=color_palette) + labs(x="Temporal stability", y="AUC") + theme(legend.position = 'top',
                                                                                                                                         legend.key.size = unit(1.5, 'lines')) +
  scale_x_continuous(breaks = trans_breaks(identity, identity, n = 3))

dev.off()


# brier scores
setwd("~/Repos/predictive-monitoring-thesis/results_stability_brier/")
files <- list.files()
files

data <- data.frame()
for (file in files) {
  tmp <- read.table(file, sep=";", header=TRUE)
  data <- rbind(data, tmp)
}

head(subset(data, method=="prefix_index"), 43)

data[grepl("lstm", data$cls), "Method"] <- "LSTM"
data[grepl("rf", data$cls) & data$method=="single_index", "Method"] <- "RF_idx_pad"
data[grepl("rf", data$cls) & data$method=="prefix_index", "Method"] <- "RF_idx_mul"
data[grepl("rf", data$cls) & data$method=="single_agg", "Method"] <- "RF_agg"
data[grepl("xgboost", data$cls) & data$method=="single_index", "Method"] <- "XGB_idx_pad"
data[grepl("xgboost", data$cls) & data$method=="prefix_index", "Method"] <- "XGB_idx_mul"
data[grepl("xgboost", data$cls) & data$method=="single_agg", "Method"] <- "XGB_agg"

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

datasets1 <- c("bpic2011_1", "bpic2011_2", "bpic2011_3", "bpic2011_4", "bpic2015_1", "bpic2015_2", "bpic2015_3", 
               "bpic2015_4", "bpic2015_5", "production", "insurance_1", "insurance_2")
datasets2 <- c("sepsis_1", "sepsis_2", "sepsis_3", "bpic2012_1", "bpic2012_2", "bpic2012_3", "bpic2017_1", 
               "bpic2017_2", "bpic2017_3", "traffic", "hospital_1", "hospital_2", "DR", "github", "LtC")

library(RColorBrewer)

data_cal <- subset(data, grepl("calibrated", cls))
data_uncal <- subset(data, !grepl("calibrated", cls))

data_merged <- merge(data_cal[,c("dataset", "nr_events", "Method", "score")], data_uncal[,c("dataset", "nr_events", "Method", "score")], by=c("dataset", "nr_events", "Method"))

head(data_merged)

data_merged$brier_diff <- data_merged$score.y - data_merged$score.x
head(data_merged)

color_palette <- c("#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")[c(1,8,4,6,2,7,3)]
png("/home/irene/Dropbox/phd_thesis_irene/stability_paper/brier_3cols_1.png", width=width, height=height)
ggplot(subset(data_merged, dataset %in% datasets1), aes(x=as.numeric(nr_events), y=brier_diff, color=Method, shape=Method)) + 
  geom_point(size=point_size) + geom_line(size=line_size) + 
  theme_bw(base_size=base_size) + facet_wrap(~dataset, scales="free", ncol=3) + scale_color_manual(values=color_palette) + 
  labs(x="Prefix length", y="Brier(uncal) - Brier(cal)") + theme(legend.position = 'top', legend.key.size = unit(1.5, 'lines')) + geom_hline(yintercept=0)
dev.off()

png("/home/irene/Dropbox/phd_thesis_irene/stability_paper/brier_3cols_2.png", width=width, height=height*5/4)
ggplot(subset(data_merged, dataset %in% datasets2), aes(x=as.numeric(nr_events), y=brier_diff, color=Method, shape=Method)) + 
  geom_point(size=point_size) + geom_line(size=line_size) + 
  theme_bw(base_size=base_size) + facet_wrap(~dataset, scales="free", ncol=3) + scale_color_manual(values=color_palette) + 
  labs(x="Prefix length", y="Brier(uncal) - Brier(cal)") + theme(legend.position = 'top', legend.key.size = unit(1.5, 'lines')) + geom_hline(yintercept=0)
dev.off()

