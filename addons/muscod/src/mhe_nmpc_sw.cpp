// *****************************************************************************
// Includes
// *****************************************************************************

#include <iostream>
#include <dlfcn.h>
#include <sys/stat.h>

// GRL
#include <grl/policies/mhe_nmpc_sw.h>

// MUSCOD-II interface
#include <wrapper.hpp>

using namespace grl;

REGISTER_CONFIGURABLE(MHE_NMPC_SWPolicy)

MHE_NMPC_SWPolicy::~MHE_NMPC_SWPolicy()
{
  safe_delete(&muscod_mhe_);
  safe_delete(&muscod_nmpc_);
}

void MHE_NMPC_SWPolicy::request(ConfigurationRequest *config)
{
  config->push_back(CRP("model_name", "Name of the model in grl", model_name_));
  config->push_back(CRP("mhe_model_name", "Name of MUSCOD MHE model library", mhe_model_name_));
  config->push_back(CRP("nmpc_model_name", "Name of MUSCOD MHE model library", nmpc_model_name_));
  config->push_back(CRP("outputs", "int.action_dims", "Number of outputs", (int)outputs_, CRP::System, 1));
  config->push_back(CRP("verbose", "Verbose mode", (int)verbose_, CRP::System, 0, 1));
}

void *MHE_NMPC_SWPolicy::setup_model_path(const std::string path, const std::string model, const std::string lua_model)
{
  // get the library handle,
  std::string so_path  = path + "/" + "lib" + model + ".so";
  void *so_handle = dlopen(so_path.c_str(), RTLD_NOW|RTLD_GLOBAL);
  if (so_handle==NULL)
  {
    std::cout << "ERROR: Could not load MUSCOD-II shared model library: '" << so_path << "'" << std::endl;
    std::cout << "dlerror responce: " << dlerror() << std::endl;
    std::cout << "bailing out ..." << std::endl;
    exit(EXIT_FAILURE);
  }

  // get the function handle
  void (*so_set_path)(std::string, std::string);
  std::string so_set_path_fn = "set_path"; // name of a function which sets the path
  so_set_path = (void (*)(std::string, std::string)) dlsym(so_handle, so_set_path_fn.c_str());
  if (so_set_path==NULL)
  {
    std::cout << "ERROR: Could not symbol in shared library: '" << so_set_path_fn << "'" << std::endl;
    std::cout << "bailing out ..." << std::endl;
    std::exit(-1);
  }

  // ... and finally set the paths
  if (verbose_)
  {
    std::cout << "MUSCOD: setting new problem path to: '" << path << "'" <<std::endl;
    std::cout << "MUSCOD: setting new Lua model file to: '" << lua_model << "'" <<std::endl;
  }
  so_set_path(path, lua_model);

  return so_handle;
}

void MHE_NMPC_SWPolicy::configure(Configuration &config)
{
  std::string model_path;
  model_path        = std::string(MUSCOD_CONFIG_DIR);
  model_name_       = config["model_name"].str();
  mhe_model_name_   = config["mhe_model_name"].str();
  nmpc_model_name_  = config["nmpc_model_name"].str();
  outputs_          = config["outputs"];
  verbose_          = config["verbose"];

  // Setup path for the problem description library and lua, csv, dat files used by it
  std::string problem_path  = model_path + "/" + model_name_;

  //----------------- Set path in the problem description library ----------------- //
  setup_model_path(problem_path, mhe_model_name_, "");
  void * so_handle_nmpc = setup_model_path(problem_path, nmpc_model_name_, "");

  //------------------- Initialize MHE ------------------- //
  muscod_mhe_ = new MUSCOD();
  mhe_ = new MHEProblem(problem_path.c_str(), mhe_model_name_.c_str(), muscod_mhe_);

  Vector hs(mhe_->NXD() + mhe_->NU()), ss(mhe_->NXD() + mhe_->NU()), qs(mhe_->NU());
  hs << 0.00, 3.14159, 0.00, 0.00, 0.00; // hanging down
  ss << 1.00, 1.00,    1.00, 1.00, 0.10; // no error
  qs << 0.00; // no control

  // initialize measurement horizon with data
  mhe_->initialize_horizon(hs, ss, qs);

  //------------------- Initialize NMPC ------------------- //
  muscod_nmpc_ = new MUSCOD();
  nmpc_ = new NMPCProblem(problem_path.c_str(), nmpc_model_name_.c_str(), muscod_nmpc_);

  // Allocate memory
  initial_sd_ = VectorConstructorFill(nmpc_->NXD(), 0);
  initial_pf_ = VectorConstructorFill(nmpc_->NP(), 0);
  initial_qc_ = VectorConstructorFill(nmpc_->NU(), 0);
  final_sd_   = VectorConstructorFill(nmpc_->NXD(), 0);

  hs_ = VectorConstructorFill(mhe_->NXD() + mhe_->NU(), 0);
  ss_ = VectorConstructorFill(mhe_->NXD() + mhe_->NU(), 1);
  ss_ << 1.00, 1.00, 1.00, 1.00, 0.10; // no error

  // FIXME This part is needed
  // if (!verbose_) {
    // muscod_mhe_->setLogLevelScreen(-1);
    // muscod_mhe_->setLogLevelAndFile(-1, NULL, NULL);
    // muscod_nmpc_->setLogLevelScreen(-1);
    // muscod_nmpc_->setLogLevelAndFile(-1, NULL, NULL);
  // }

/*
  // save solver state
  // FIXME this part is needed
  data_.backup_muscod_state(muscod_);
  data_.sd = ConstantVector(data_.NXD, 0.0);
  data_.pf = ConstantVector(data_.NP,  0.0);
*/

  if (verbose_)
    std::cout << "MUSCOD is ready!" << std::endl;
}

void MHE_NMPC_SWPolicy::reconfigure(const Configuration &config)
{
}


void MHE_NMPC_SWPolicy::muscod_reset(const Vector &initial_obs, double time)
{
  // FIXME
  // load solution state
  // data_.restore_muscod_state(muscod_);

  // // Reinitialize state and time
  // for (int IP = 0; IP < data_.NP; ++IP)
  //   data_.pf[IP] = time;

  // initialize NMPC
  for (int inmpc = 0; inmpc < 10; ++inmpc) {
    // 1) Feedback: Embed parameters and initial value from MHE
    nmpc_->feedback(initial_obs, initial_pf_, &initial_qc_);
    // 2) Transition
    nmpc_->transition();
    // 3) Preparation
    nmpc_->preparation();
  }

  // initialize MHE
  for (int imhe = 0; imhe < 10; ++imhe) {
    // 1) Feedback
    mhe_->feedback();
    // 2) Transition
    mhe_->transition();
    // 3) Preparation
    mhe_->preparation();
  }

  if (verbose_)
    std::cout << "MUSCOD is reseted!" << std::endl;
}

MHE_NMPC_SWPolicy *MHE_NMPC_SWPolicy::clone() const
{
  return NULL;
}

void MHE_NMPC_SWPolicy::act(double time, const Vector &in, Vector *out)
{
  if (time == 0.0)
  {
    muscod_reset(in, time);
    initial_sd_ << in;
    initial_pf_ << 0.0;
    initial_qc_ << 0.0;
  }

  if (verbose_)
    std::cout << "time: [ " << time << " ]; state: [ " << in << "]" << std::endl;

  out->resize(outputs_);

  // Run mutiple MHE iterations
  const unsigned int nmhe = 10;
  for (int imhe = 0; imhe < nmhe; ++imhe) {
    // NOTE compose and inject measurement only at the first iteration of MHE
    if (nmhe > 0 && imhe == 0) {
      // 0) Compose new measurement
      // NOTE measurement consists of simulation result + feedback control
      // m_hs = [ xd[0], ..., xd[NXD-1], u[0], ..., u[NU-1] ]
      hs_ << in, VectorConstructorFill(mhe_->NU(), 0);
      // std::cout << "new_measurement = " << hs_ << std::endl;
      // 1) Inject measurements
      mhe_->inject_measurement(hs_, ss_, initial_qc_);
      // mhe_->print_horizon();
    }
    // 2) Feedback
    mhe_->feedback();

    // 4) Shifting?
    // NOTE do that only once at last iteration
    // NOTE this has to be done before the transition phase
    if (nmhe > 0 && imhe == nmhe-1) {
      mhe_->shifting(1);
    }

    // 3) Transition
    // NOTE: states and parameters are only updated after transition phase
    mhe_->transition();

    // 3) Get parameters and state of last shooting node
    // NOTE do that only once at last iteration
    if (nmhe > 0 && imhe == nmhe-1) {
      mhe_->get_initial_sd_and_pf(&initial_sd_, &initial_pf_);
    }

    // 6) Preparation
    mhe_->preparation();
  }

  // Run multiple NMPC iterations
  const unsigned int nnmpc = 10;
  for (int inmpc = 0; inmpc < nnmpc; ++inmpc) {
    // 1) Feedback: Embed parameters and initial value from MHE
    // NOTE the same initial values (sd, pf) are embedded several time,
    //      but this will result in the same solution as running a MUSCOD
    //      instance for several iterations
    // initial_sd_ << obs;
    // initial_pf_ << 0.0;
    nmpc_->feedback(initial_sd_, initial_pf_, &initial_qc_);
    // 2) Shifting
    // NOTE do that only once at last iteration
    // NOTE this has to be done before the transition phase
    if (nnmpc > 0 && inmpc == nnmpc-1) {
      nmpc_->shifting(1);
    }
    // 3) Transition
    nmpc_->transition();
    // 4) Preparation
    nmpc_->preparation();
  }

  // Here we can return the feedback control
  (*out) = initial_qc_;

  if (verbose_)
    std::cout << "Feedback Control: [" << *out << "]" << std::endl;
}
