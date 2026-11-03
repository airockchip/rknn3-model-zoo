/* Laya text request demo. */
#include <stdio.h>
#include <string.h>
#include <limits.h>

#include "laya_text.h"

static json display_options(const json &question, const json &answer)
{
	json options = json::object();
	json criteria = question.value("criteria", json());
	const char *kind = laya_json_string(answer["type"]);

	if (!strcmp(kind, "score"))
		return answer["legend"];
	if (!strcmp(kind, "noul")) {
		options = {{"false", "no, the statement does not hold"},
		    {"true", "yes, the statement holds"}};
	} else {
		for (json::const_iterator item = answer["probabilities"].begin();
		     item != answer["probabilities"].end(); ++item)
			options[item.key()] = "";
	}
	if (criteria.is_object()) {
		for (json::iterator item = options.begin(); item != options.end(); ++item) {
			if (criteria.contains(item.key()) && criteria[item.key()].is_string() &&
			    laya_json_string(criteria[item.key()])[0])
				item.value() = criteria[item.key()];
		}
	}
	return options;
}

/* Use English display labels; task text is printed in its original language. */
static void print_results(const json &request, const json &answers, const json &labels,
    const char *output_path)
{
	printf("\nInput:\n  %s\n", laya_json_string(request["state"]));
	for (json::const_iterator item = request["questions"].begin();
	     item != request["questions"].end(); ++item) {
		const json &question = item.value();
		const json &answer = answers[item.key()];
		json options = display_options(question, answer);
		const char *label = laya_json_string(labels[item.key()]);
		const char *description = laya_json_string(options[label]);
		double probability;

		printf("\nQuestion [%s]: %s\n", item.key().c_str(),
		    laya_json_string(question["instructions"]));
		for (json::const_iterator option = options.begin(); option != options.end(); ++option) {
			const char *text = laya_json_string(option.value());
			printf("  %s%s%s\n", option.key().c_str(),
			    text[0] ? ": " : "", text);
		}
		if (answer.contains("noul")) {
			double positive = answer["noul"].get<double>();
			probability = !strcmp(label, "true") ? positive : 1.0 - positive;
		} else {
			probability = answer["probabilities"][label].get<double>();
		}
		printf("Result: %s", label);
		if (description[0])
			printf(" (%s)", description);
		printf(", probability %.2f%%\n", probability * 100.0);
		if (answer.contains("score"))
			printf("Score: %.4f\n", answer["score"].get<double>());
		if (answer.contains("noul"))
			printf("Positive probability: %.2f%%\n", answer["noul"].get<double>() * 100.0);
	}
	if (output_path[0])
		printf("\nFull result saved to: %s\n", output_path);
}

static void print_usage(const char *program)
{
	fprintf(stderr,
	    "Usage: %s --model_path DIR --task TASK --text TEXT [--output_path JSON]\n"
	    "Tasks: customer_service, review_analysis\n", program);
}

/* Combine the task's questions with this invocation's text input. */
static int build_request(const char *model_path, const char *task, const char *text,
    json *request)
{
	char path[PATH_MAX];
	const char *filename;
	json preset;

	if (!strcmp(task, "customer_service"))
		filename = "customer_service.json";
	else if (!strcmp(task, "review_analysis"))
		filename = "review_analysis.json";
	else {
		fprintf(stderr, "Unknown task: %s\n", task);
		return -1;
	}
	if (strspn(text, " \t\r\n") == strlen(text)) {
		fprintf(stderr, "%s requires non-empty --text\n", task);
		return -1;
	}
	if (laya_join_path(path, model_path, filename) < 0)
		return -1;
	if (laya_load_json(path, &preset) < 0 || !preset.is_object() ||
	    !preset.contains("questions") || !preset["questions"].is_object() ||
	    preset["questions"].empty()) {
		fprintf(stderr, "Invalid task configuration: %s\n", path);
		return -1;
	}
	*request = {{"state", text}, {"questions", preset["questions"]}};
	return 0;
}

int main(int argc, char **argv)
{
	const char *model_path = "", *task = "", *text = "", *output_path = "";

	if (argc == 2 && !strcmp(argv[1], "--help")) {
		print_usage(argv[0]);
		return 0;
	}
	for (int i = 1; i < argc; i += 2) {
		const char **value = NULL;

		if (!strcmp(argv[i], "--model_path"))
			value = &model_path;
		else if (!strcmp(argv[i], "--task"))
			value = &task;
		else if (!strcmp(argv[i], "--text"))
			value = &text;
		else if (!strcmp(argv[i], "--output_path"))
			value = &output_path;
		if (!value || i + 1 >= argc || (*value)[0] || !argv[i + 1][0]) {
			fprintf(stderr, "Invalid argument: %s\n", argv[i]);
			print_usage(argv[0]);
			return 1;
		}
		*value = argv[i + 1];
	}
	if (!model_path[0] || !task[0]) {
		print_usage(argv[0]);
		return 1;
	}
	struct laya_context ctx = {0};
	/* Cleanup also handles failures before any model resource is allocated. */
	ctx.embedding_fd = -1;
	json config, request, answers = json::object(), labels = json::object();
	int ret = -1, tokens_total = 0;
	if (build_request(model_path, task, text, &request) < 0)
		goto cleanup;
	if (laya_runtime_init(&ctx, model_path) < 0 ||
	    laya_load_resources(&ctx, model_path, &config) < 0) {
		fprintf(stderr, "Failed to initialize Laya resources/model\n");
		goto cleanup;
	}
	for (json::const_iterator item = request["questions"].begin();
	     item != request["questions"].end(); ++item) {
		json keys, answer;
		int tokens = 0, qtype = 0;
		if (laya_prepare(
			&ctx, request["state"], item.value(), &keys, &tokens, &qtype) < 0 ||
		    laya_runtime_run(&ctx) < 0 ||
		    laya_postprocess(&ctx, item.value(), keys, config, qtype, &answer) < 0) {
			fprintf(stderr,
			    "Question failed: %s (check type, criteria and token capacity)\n",
			    item.key().c_str());
			goto cleanup;
		}
		answers[item.key()] = answer;
		/* Select before display rounding, including score and noul questions. */
		const float16 *logits = (const float16 *)ctx.tensors[LAYA_OUTPUT_LOGITS].mem->virt_addr;
		size_t best = 0;
		for (size_t i = 1; i < keys.size(); ++i) {
			if (fp16_to_fp32(logits[i]) > fp16_to_fp32(logits[best]))
				best = i;
		}
		labels[item.key()] = keys[best];
		tokens_total += tokens;
	}
	{
		json result = {{"model", "rl-agent"}, {"answers", answers},
		    {"usage", {{"input_tokens", tokens_total}, {"output_tokens", 0}}}};
		ret = output_path[0] ? laya_save_json(output_path, &result) : 0;
		if (ret == 0)
			print_results(request, answers, labels, output_path);
		else
			fprintf(stderr, "Cannot write output file: %s\n", output_path);
	}
cleanup:
	if (laya_release(&ctx) < 0)
		ret = -1;
	return ret == 0 ? 0 : 1;
}
