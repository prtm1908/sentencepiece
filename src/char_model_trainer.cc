// Include LevelDB headers
#include <leveldb/db.h>
#include <string>
#include <stdexcept>
#include <cmath>
#include "char_model.h"
#include "char_model_trainer.h"
#include "util.h"

namespace sentencepiece {
namespace character {

util::Status Trainer::Train() {
  RETURN_IF_ERROR(status());

  CHECK_OR_RETURN(normalizer_spec_.escape_whitespaces());
  CHECK_EQ_OR_RETURN(TrainerSpec::CHAR, trainer_spec_.model_type());

  RETURN_IF_ERROR(LoadSentences());

  const int vocab_size = trainer_spec_.vocab_size() - meta_pieces_.size();
  CHECK_GE_OR_RETURN(vocab_size, 0);

  uint64 sum = 0;
  
  // LevelDB setup
  leveldb::DB* db;
  leveldb::Options options;
  options.create_if_missing = true;
  leveldb::Status status = leveldb::DB::Open(options, "required_chars_db", &db);
  if (!status.ok()) {
    throw std::runtime_error("Unable to open/create database: required_chars_db");
  }

  // Iterate through required_chars_ stored in LevelDB
  leveldb::Iterator* it = db->NewIterator(leveldb::ReadOptions());
  for (it->SeekToFirst(); it->Valid(); it->Next()) {
    sum += std::stoull(it->value().ToString());
  }
  delete it;

  const auto logsum = std::log(static_cast<float>(sum));

  CHECK_OR_RETURN(final_pieces_.empty());
  it = db->NewIterator(leveldb::ReadOptions());
  for (it->SeekToFirst(); it->Valid(); it->Next()) {
    if (!trainer_spec_.use_all_vocab() && final_pieces_.size() == static_cast<size_t>(vocab_size)) {
      break;
    }
    final_pieces_.emplace_back(
        string_util::UnicodeCharToUTF8(std::stoi(it->key().ToString())),
        std::log(static_cast<float>(std::stoull(it->value().ToString()))) - logsum);
  }
  delete it;

  if (trainer_spec_.use_all_vocab()) {
    trainer_spec_.set_vocab_size(final_pieces_.size() + meta_pieces_.size());
  }

  // Close the LevelDB
  delete db;

  return Save();
}

}  // namespace character
}  // namespace sentencepiece
