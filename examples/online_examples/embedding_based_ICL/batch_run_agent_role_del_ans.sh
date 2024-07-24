python examples/online_examples/embedding_based_ICL/batch_run_role_followups.py \
--online_config_pth conf/agent_followup_configs/online_icl_config/online_icl_config_ver_del_ans.yaml \
--data_pth "data/app_data/agent_followup_data/real_conversation_data/智能体追问/session_role_conversation_gf_ver_2024-07-18 19:57:54.json" \
--sav_dir data/app_data/agent_followup_data/real_conversation_data/智能体追问/results \
--prefix role_followups_gf_del_ans \
--query_only 1



python examples/online_examples/embedding_based_ICL/batch_run_role_followups.py \
--online_config_pth conf/agent_followup_configs/online_icl_config/online_icl_config_ver_del_ans.yaml \
--data_pth "data/app_data/agent_followup_data/real_conversation_data/智能体追问/session_tool_conversation_ver_2024-07-18 19:59:16.json" \
--sav_dir data/app_data/agent_followup_data/real_conversation_data/智能体追问/results \
--prefix tool_followups_del_ans \
--query_only 1

echo "========================================"


python examples/online_examples/embedding_based_ICL/batch_run_role_followups.py \
--online_config_pth conf/agent_followup_configs/online_icl_config/online_icl_config_ver_del_ans.yaml \
--data_pth "data/app_data/agent_followup_data/real_conversation_data/智能体追问/session_role_conversation_ver_2024-07-10 11:58:00.json" \
--sav_dir data/app_data/agent_followup_data/real_conversation_data/智能体追问/results \
--prefix role_followups_del_ans \
--query_only 1

echo "========================================"


python examples/online_examples/embedding_based_ICL/batch_run_role_followups.py \
--online_config_pth conf/agent_followup_configs/online_icl_config/online_icl_config1.yaml \
--data_pth "data/app_data/agent_followup_data/real_conversation_data/智能体追问/session_role_conversation_gf_ver_2024-07-18 19:57:54.json" \
--sav_dir data/app_data/agent_followup_data/real_conversation_data/智能体追问/results \
--prefix role_followups_gf \
--query_only 0


echo "========================================"

python examples/online_examples/embedding_based_ICL/batch_run_role_followups.py \
--online_config_pth conf/agent_followup_configs/online_icl_config/online_icl_config1.yaml \
--data_pth "data/app_data/agent_followup_data/real_conversation_data/智能体追问/session_role_conversation_ver_2024-07-10 11:58:00.json" \
--sav_dir data/app_data/agent_followup_data/real_conversation_data/智能体追问/results \
--prefix role_followups \
--query_only 0

echo "========================================"


python examples/online_examples/embedding_based_ICL/batch_run_role_followups.py \
--online_config_pth conf/agent_followup_configs/online_icl_config/online_icl_config1.yaml \
--data_pth "data/app_data/agent_followup_data/real_conversation_data/智能体追问/session_tool_conversation_ver_2024-07-18 19:59:16.json" \
--sav_dir data/app_data/agent_followup_data/real_conversation_data/智能体追问/results \
--prefix tool_followups \
--query_only 0