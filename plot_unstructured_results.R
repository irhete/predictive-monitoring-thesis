library(ggplot2)
library(scmamp)
library(reshape2)

setwd("~/Repos/predictive-monitoring-thesis/results_unstructured/")
files <- list.files()
files

data <- data.frame()
for (file in files) {
  tmp <- read.table(file, sep=";", header=TRUE)
  if (!grepl("bong", file) & !grepl("nb", file) & !grepl("lda", file) & !grepl("pv", file)) {
    tmp$text_method_enc <- "no_text"
    names(tmp)[names(tmp) == "method"] <- "bucket_enc"
  }
  data <- rbind(data, tmp)
}

data$dataset <- as.character(data$dataset)
data[data$dataset=="dc", "dataset"] <- "DR"
data[data$dataset=="crm2", "dataset"] <- "LtC"
data[data$dataset=="github", "dataset"] <- "Github"

data$bucket_enc <- as.character(data$bucket_enc)
data[data$text_method_enc=="nb_laststate", "bucket_enc"] <- paste(data[data$text_method_enc=="nb_laststate", "bucket_enc"], "laststate", sep="_")
data[data$text_method_enc=="pv_laststate", "bucket_enc"] <- paste(data[data$text_method_enc=="pv_laststate", "bucket_enc"], "laststate", sep="_")
data[data$text_method_enc=="lda_laststate", "bucket_enc"] <- paste(data[data$text_method_enc=="lda_laststate", "bucket_enc"], "laststate", sep="_")
data[data$text_method_enc=="bong_laststate", "bucket_enc"] <- paste(data[data$text_method_enc=="bong_laststate", "bucket_enc"], "laststate", sep="_")
data[data$text_method_enc=="nb_agg", "bucket_enc"] <- paste(data[data$text_method_enc=="nb_agg", "bucket_enc"], "agg", sep="_")
data[data$text_method_enc=="pv_agg", "bucket_enc"] <- paste(data[data$text_method_enc=="pv_agg", "bucket_enc"], "agg", sep="_")
data[data$text_method_enc=="lda_agg", "bucket_enc"] <- paste(data[data$text_method_enc=="lda_agg", "bucket_enc"], "agg", sep="_")
data[data$text_method_enc=="bong_agg", "bucket_enc"] <- paste(data[data$text_method_enc=="bong_agg", "bucket_enc"], "agg", sep="_")
data$text_method <- "No text"
data[grepl("nb", data$text_method_enc), "text_method"] <- "NB"
data[grepl("bong", data$text_method_enc), "text_method"] <- "BoNG"
data[grepl("lda", data$text_method_enc), "text_method"] <- "LDA"
data[grepl("pv", data$text_method_enc), "text_method"] <- "PV"
tmp <- subset(data, bucket_enc == "prefix_index")
tmp$bucket_enc <- "prefix_index_laststate"
data <- subset(data, bucket_enc != "prefix_index")
data <- rbind(data, tmp)
tmp$bucket_enc <- "prefix_index_agg"
data <- rbind(data, tmp)

data$method <- paste(data$cls, data$bucket_enc, data$text_method_enc, sep="_")
data <- subset(data, dataset != "DR" | nr_events <= 6)

head(data)

cbbPalette <- c("#0072B2", "#CC79A7", "#009E73", "#000000", "#E69F00")
base_size = 28
line_size = 0.5
point_size = 2

# by classifier
ggplot(subset(data, metric=="auc" & nr_events != -1 & bucket_enc=="prefix_index_agg" & text_method!="No text"), aes(x=nr_events, y=score, color=cls, shape=cls)) + 
  geom_point(size=point_size) + geom_line(size=line_size) +
  theme_bw(base_size=base_size) + facet_wrap(text_method~dataset, scales="free", ncol=3) + theme(legend.position="top") + scale_color_manual(values=cbbPalette)

# RF
png("~/Dropbox/phd_thesis_irene/unstructured_paper/auc_all_rf.png", width=1000, height=1250)
ggplot(subset(data, metric=="auc" & nr_events != -1 & cls=="rf"), aes(x=nr_events, y=score, color=text_method, shape=text_method)) + 
  geom_point(size=point_size) + geom_line(size=line_size) +
  theme_bw(base_size=base_size) + facet_wrap(bucket_enc~dataset, scales="free", ncol=3) + theme(legend.position="top") + scale_color_manual(values=cbbPalette)
dev.off()

png("~/Dropbox/phd_thesis_irene/unstructured_paper/auc_all_xgboost.png", width=1000, height=1250)
ggplot(subset(data, metric=="auc" & nr_events != -1 & cls=="xgboost"), aes(x=nr_events, y=score, color=text_method, shape=text_method)) + 
  geom_point(size=point_size) + labs(x="Prefix length", y="AUC") + geom_line(size=line_size) +
  theme_bw(base_size=base_size) + facet_wrap(bucket_enc~dataset, scales="free", ncol=3) + theme(legend.position="top") + scale_color_manual(values=cbbPalette)
dev.off()

png("~/Dropbox/phd_thesis_irene/unstructured_paper/auc_all_xgboost_flipped.png", width=1000*4/3, height=1250*3/4)
ggplot(subset(data, metric=="auc" & nr_events != -1 & cls=="xgboost"), aes(x=nr_events, y=score, color=text_method, shape=text_method)) + 
  geom_point(size=point_size) + geom_line(size=line_size) + geom_line(size=line_size) + labs(x="Prefix length", y="AUC") +
  theme_bw(base_size=base_size) + facet_wrap(dataset~bucket_enc, scales="free", ncol=4) + theme(legend.position="top") + scale_color_manual(values=cbbPalette)
dev.off()

png("~/Dropbox/phd_thesis_irene/unstructured_paper/auc_all_xgboost.png", width=1000, height=1250)
ggplot(subset(data, metric=="auc" & nr_events != -1 & cls=="xgboost"), aes(x=nr_events, y=score, color=text_method, shape=text_method)) + 
  geom_point(size=point_size) + geom_line(size=line_size) +
  theme_bw(base_size=base_size) + facet_wrap(bucket_enc~dataset, scales="free", ncol=3) + theme(legend.position="top") + scale_color_manual(values=cbbPalette)
dev.off()

png("~/Dropbox/phd_thesis_irene/unstructured_paper/auc_all_logit.png", width=1000, height=1250)
ggplot(subset(data, metric=="auc" & nr_events != -1 & cls=="logit"), aes(x=nr_events, y=score, color=text_method, shape=text_method)) + 
  geom_point(size=point_size) + geom_line(size=line_size) +
  theme_bw(base_size=base_size) + facet_wrap(bucket_enc~dataset, scales="free", ncol=3) + theme(legend.position="top") + scale_color_manual(values=cbbPalette)
dev.off()


ggplot(subset(data, metric=="auc" & cls=="xgboost" & text_method_enc=="no_text" & nr_events != -1), aes(x=nr_events, y=score, color=bucket_enc)) + geom_point() + geom_line() +
  theme_bw(base_size=26) + facet_wrap(.~dataset, scales="free", ncol=3)

png("~/Dropbox/phd_thesis_irene/unstructured_paper/auc_bong_rf.png", width=1000, height=400)
ggplot(subset(data, metric=="auc" & text_method=="BoNG" & nr_events != -1 & cls=="rf"), aes(x=nr_events, y=score, color=bucket_enc, shape=bucket_enc)) + geom_point(size=point_size) + geom_line(size=line_size) +
  theme_bw(base_size=26) + facet_wrap(~dataset, scales="free", ncol=3) + theme(legend.position="top") + scale_color_manual(values=cbbPalette)
dev.off()

png("~/Dropbox/phd_thesis_irene/unstructured_paper/auc_register_rf.png", width=900, height=500)
ggplot(subset(data, metric=="auc" & nr_events != -1 & dataset%in%c("LtC", "DR") & bucket_enc=="single_laststate"), aes(x=nr_events, y=score, color=text_method, shape=text_method)) + 
  geom_point(size=point_size) + geom_line(size=line_size) +
  theme_bw(base_size=base_size) + facet_wrap(bucket_enc~dataset, scales="free", ncol=3) + theme(legend.position="top") + scale_color_manual(values=cbbPalette)
dev.off()
