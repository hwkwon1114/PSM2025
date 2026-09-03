//
//  main.cpp
//  Elasticity
//
//  Created by Wim van Rees on 2/15/16. 
//  Modified by Vladislav Sushitskii on 03/29/22.
//  Modified by Putong Kang on 12/10/25.
//  Copyright © 2025 Wim van Rees, Vladislav Sushitskii and Putong Kang. All rights reserved.
//

#include "common.hpp"
#include "ArgumentParser.hpp"
#include "Sim.hpp"

#include "Sim_Bilayer_Growth.hpp"
// Added functionality for creating output directories if they do not exist
#include <filesystem>
namespace fs = std::filesystem;

#ifdef USETBB
#include "tbb/task_scheduler_init.h"
#endif

int main(int argc,  const char ** argv)
{
#ifdef USETBB
    const int num_threads = tbb::task_scheduler_init::default_num_threads();
    tbb::task_scheduler_init init(num_threads);
    std::cout << "Starting TBB with " << num_threads << " threads " << std::endl;
#endif

    BaseSim * sim = nullptr;

    ArgumentParser parser(argc,argv);

    // ---- out_dir: create and switch working directory early ----
    const std::string out_dir_str = parser.parse<std::string>("-out_dir", ".");
    if (!out_dir_str.empty() && out_dir_str != ".")
    {
        fs::path base = fs::current_path();     // where you launched ./shell
        fs::path outp(out_dir_str);
        if (!outp.is_absolute()) outp = base / outp;

        fs::create_directories(outp);
        fs::current_path(outp);

        std::cout << "[I] out_dir = " << fs::current_path() << std::endl;
    }
    // -----------------------------------------------------------

    const std::string simCase = parser.parse<std::string>("-sim", "");

    if(simCase == "bilayer_growth")
        sim = new Sim_Bilayer_Growth(parser);
    else
    {
        std::cout << "No valid sim case defined. Options are \n";
        std::cout << "\t -sim bilayer_growth\n";
        helpers::catastrophe("sim case does not exist",__FILE__,__LINE__);
    }

    sim->init();
    sim->run();

    delete sim;

    parser.finalize();

    return 0;
}
