#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cstring>
#include <sstream>
#include <algorithm>
#include "varchar.h"

using namespace std;

vector<int> c_id;
vector<int> l_id;
vector<int> o_id;
vector<int> p_id;
vector<int> s_id;
vector<int> n_id;
vector<int> r_id;
vector<int> ps_id; 

// Customer Table
vector<int> c_custkey;
vector<VarChar<25>> c_name;
vector<VarChar<40>> c_address;
vector<int> c_nationkey;
vector<VarChar<15>> c_phone;
vector<double> c_acctbal;
vector<VarChar<10>> c_mktsegment;
vector<VarChar<117>> c_comment;

// Lineitem Table
vector<int> l_orderkey;
vector<int> l_partkey;
vector<int> l_suppkey;
vector<int> l_linenumber;
vector<double> l_quantity;
vector<double> l_extendedprice;
vector<double> l_discount;
vector<double> l_tax;
vector<VarChar<1>> l_returnflag;
vector<VarChar<1>> l_linestatus;
vector<int> l_shipdate;
vector<int> l_commitdate;
vector<int> l_receiptdate;
vector<VarChar<25>> l_shipinstruct;
vector<VarChar<10>> l_shipmode;
vector<VarChar<44>> l_comment;

// Nation Table
vector<int> n_nationkey;
vector<VarChar<25>> n_name;
vector<int> n_regionkey;
vector<VarChar<152>> n_comment;

// Orders Table
vector<int> o_orderkey;
vector<int> o_custkey;
vector<VarChar<1>> o_orderstatus;
vector<double> o_totalprice;
vector<int> o_orderdate;
vector<VarChar<15>> o_orderpriority;
vector<VarChar<15>> o_clerk;
vector<int> o_shippriority;
vector<VarChar<79>> o_comment;

// Part Table
vector<int> p_partkey;
vector<VarChar<55>> p_name;
vector<VarChar<25>> p_mfgr;
vector<VarChar<10>> p_brand;
vector<VarChar<25>> p_type;
vector<int> p_size;
vector<VarChar<10>> p_container;
vector<double> p_retailprice;
vector<VarChar<23>> p_comment;

// Partsupp Table
vector<int> ps_partkey;
vector<int> ps_suppkey;
vector<int> ps_availqty;
vector<double> ps_supplycost;
vector<VarChar<199>> ps_comment;

// Region Table
vector<int> r_regionkey;
vector<VarChar<25>> r_name;
vector<VarChar<152>> r_comment;

// Supplier Table
vector<int> s_suppkey;
vector<VarChar<25>> s_name;
vector<VarChar<40>> s_address;
vector<int> s_nationkey;
vector<VarChar<15>> s_phone;
vector<double> s_acctbal;
vector<VarChar<101>> s_comment;


void load_lineitem(string path, bool verbose=false)
{
    string dataset_name = "lineitem.tbl";
    ifstream in(path+dataset_name);
    if(!in)
    {
        cerr << "Cannot open the File : " << path << endl;
    }
    string line;
    string token;
    while(getline(in, line))
    {
        stringstream ss(line);
        getline(ss, token, '|');
        l_orderkey.push_back(stoi(token));
        getline(ss, token, '|');
        l_partkey.push_back(stoi(token));
        getline(ss, token, '|');
        l_suppkey.push_back(stoi(token));
        getline(ss, token, '|');
        l_linenumber.push_back(stoi(token));
        getline(ss, token, '|');
        l_quantity.push_back(stof(token));
        getline(ss, token, '|');
        l_extendedprice.push_back(stof(token));
        getline(ss, token, '|');
        l_discount.push_back(stof(token));
        getline(ss, token, '|');
        l_tax.push_back(stof(token));
        getline(ss, token, '|');
        l_returnflag.push_back(VarChar<1>(std::wstring(token.begin(), token.end()).c_str()));
        getline(ss, token, '|');
        l_linestatus.push_back(VarChar<1>(std::wstring(token.begin(), token.end()).c_str()));
        getline(ss, token, '|');
        token.erase(remove(token.begin(), token.end(), '-'), token.end());
        l_shipdate.push_back(stoi(token));
        getline(ss, token, '|');
        token.erase(remove(token.begin(), token.end(), '-'), token.end());
        l_commitdate.push_back(stoi(token));
        getline(ss, token, '|');
        token.erase(remove(token.begin(), token.end(), '-'), token.end());
        l_receiptdate.push_back(stoi(token));
        getline(ss, token, '|');
        l_shipinstruct.push_back(VarChar<25>(std::wstring(token.begin(), token.end()).c_str()));
        getline(ss, token, '|');
        l_shipmode.push_back(VarChar<10>(std::wstring(token.begin(), token.end()).c_str()));
        getline(ss, token, '|');
        l_comment.push_back(VarChar<44>(std::wstring(token.begin(), token.end()).c_str()));
    }

    for (int i=0; i<l_orderkey.size(); i++) l_id.push_back(i);

    if (verbose)
        for(int i=0; i<5; i++)
        {
            cout << l_orderkey[i] << " " << l_partkey[i] << " " << l_suppkey[i] << " " << l_linenumber[i] << " " << l_quantity[i] << " " << l_extendedprice[i] << " " << l_discount[i] << " " << l_tax[i] << " " << l_returnflag[i] << " " << l_linestatus[i] << " " << l_shipdate[i] << " " << l_commitdate[i] << " " << l_receiptdate[i] << " " << l_shipinstruct[i] << " " << l_shipmode[i] << " " << l_comment[i] << endl;
        }

    in.close();
    cout << "Lineitem ... Done." << endl;
}

void load_orders(string path, bool verbose=false)
{
    string dataset_name = "orders.tbl";
    ifstream in(path+dataset_name);
    if(!in)
    {
        cerr << "Cannot open the File : " << path << endl;
    }
    string line;
    string token;
    while(getline(in, line))
    {
        stringstream ss(line);
        getline(ss, token, '|');
        o_orderkey.push_back(stoi(token));
        getline(ss, token, '|');
        o_custkey.push_back(stoi(token));
        getline(ss, token, '|');
        o_orderstatus.push_back(VarChar<1>(std::wstring(token.begin(), token.end()).c_str()));
        getline(ss, token, '|');
        o_totalprice.push_back(stof(token));
        getline(ss, token, '|');
        token.erase(remove(token.begin(), token.end(), '-'), token.end());
        o_orderdate.push_back(stoi(token));
        getline(ss, token, '|');
        o_orderpriority.push_back(VarChar<15>(std::wstring(token.begin(), token.end()).c_str()));
        getline(ss, token, '|');
        o_clerk.push_back(VarChar<15>(std::wstring(token.begin(), token.end()).c_str()));
        getline(ss, token, '|');
        o_shippriority.push_back(stoi(token));
        getline(ss, token, '|');
        o_comment.push_back(VarChar<79>(std::wstring(token.begin(), token.end()).c_str()));
    }

    for (int i=0; i<o_orderkey.size(); i++) o_id.push_back(i);

    if (verbose)
        for(int i=0; i<5; i++)
        {
            cout << o_orderkey[i] << " " << o_custkey[i] << " " << o_orderstatus[i] << " " << o_totalprice[i] << " " << o_orderdate[i] << " " << o_orderpriority[i] << " " << o_clerk[i] << " " << o_shippriority[i] << " " << o_comment[i] << endl;
        }

    in.close();
    cout << "Orders ... Done." << endl;
}

void load_nation(string path, bool verbose=false)
{
    string dataset_name = "nation.tbl";
    ifstream in(path+dataset_name);
    if(!in)
    {
        cerr << "Cannot open the File : " << path << endl;
    }
    string line;
    string token;
    while(getline(in, line))
    {
        stringstream ss(line);
        getline(ss, token, '|');
        n_nationkey.push_back(stoi(token));
        getline(ss, token, '|');
        n_name.push_back(VarChar<25>(std::wstring(token.begin(), token.end()).c_str()));
        getline(ss, token, '|');
        n_regionkey.push_back(stoi(token));
        getline(ss, token, '|');
        n_comment.push_back(VarChar<152>(std::wstring(token.begin(), token.end()).c_str()));
    }

    for (int i=0; i<n_nationkey.size(); i++) n_id.push_back(i);

    if (verbose)
        for(int i=0; i<5; i++)
        {
            cout << n_nationkey[i] << " " << n_name[i] << " " << n_regionkey[i] << " " << n_comment[i] << endl;
        }

    in.close();
    cout << "Nation ... Done." << endl;
}

void load_region(string path, bool verbose=false)
{
    string dataset_name = "region.tbl";
    ifstream in(path+dataset_name);
    if(!in)
    {
        cerr << "Cannot open the File : " << path << endl;
    }
    string line;
    string token;
    while(getline(in, line))
    {
        stringstream ss(line);
        getline(ss, token, '|');
        r_regionkey.push_back(stoi(token));
        getline(ss, token, '|');
        r_name.push_back(VarChar<25>(std::wstring(token.begin(), token.end()).c_str()));
        getline(ss, token, '|');
        r_comment.push_back(VarChar<152>(std::wstring(token.begin(), token.end()).c_str()));
    }
    
    for (int i=0; i<r_regionkey.size(); i++) r_id.push_back(i);

    if (verbose)
        for(int i=0; i<5; i++)
        {
            cout << r_regionkey[i] << " " << r_name[i] << " " << r_comment[i] << endl;
        }
    
    in.close();
    cout << "Region ... Done." << endl;
}

void load_supplier(string path, bool verbose=false)
{
    string dataset_name = "supplier.tbl";
    ifstream in(path+dataset_name);
    if(!in)
    {
        cerr << "Cannot open the File : " << path << endl;
    }
    string line;
    string token;
    while(getline(in, line))
    {
        stringstream ss(line);
        getline(ss, token, '|');
        s_suppkey.push_back(stoi(token));
        getline(ss, token, '|');
        s_name.push_back(VarChar<25>(std::wstring(token.begin(), token.end()).c_str()));
        getline(ss, token, '|');
        s_address.push_back(VarChar<40>(std::wstring(token.begin(), token.end()).c_str()));
        getline(ss, token, '|');
        s_nationkey.push_back(stoi(token));
        getline(ss, token, '|');
        s_phone.push_back(VarChar<15>(std::wstring(token.begin(), token.end()).c_str()));
        getline(ss, token, '|');
        s_acctbal.push_back(stof(token));
        getline(ss, token, '|');
        s_comment.push_back(VarChar<101>(std::wstring(token.begin(), token.end()).c_str()));
    }

    for (int i=0; i<s_suppkey.size(); i++) s_id.push_back(i);

    if (verbose)
        for(int i=0; i<5; i++)
        {
            cout << s_suppkey[i] << " " << s_name[i] << " " << s_address[i] << " " << s_nationkey[i] << " " << s_phone[i] << " " << s_acctbal[i] << " " << s_comment[i] << endl;
        }

    in.close();
    cout << "Supplier ... Done." << endl;
}

void load_partsupp(string path, bool verbose=false)
{
    string dataset_name = "partsupp.tbl";
    ifstream in(path+dataset_name);
    if(!in)
    {
        cerr << "Cannot open the File : " << path << endl;
    }
    string line;
    string token;
    while(getline(in, line))
    {
        stringstream ss(line);
        getline(ss, token, '|');
        ps_partkey.push_back(stoi(token));
        getline(ss, token, '|');
        ps_suppkey.push_back(stoi(token));
        getline(ss, token, '|');
        ps_availqty.push_back(stoi(token));
        getline(ss, token, '|');
        ps_supplycost.push_back(stof(token));
        getline(ss, token, '|');
        ps_comment.push_back(VarChar<199>(std::wstring(token.begin(), token.end()).c_str()));
    }

    for (int i=0; i<ps_partkey.size(); i++) ps_id.push_back(i);

    if (verbose)
        for(int i=0; i<5; i++)
        {
            cout << ps_partkey[i] << " " << ps_suppkey[i] << " " << ps_availqty[i] << " " << ps_supplycost[i] << " " << ps_comment[i] << endl;
        }
    
    in.close();
    cout << "Partsupp ... Done." << endl;
}

void load_customer(string path, bool verbose=false)
{
    string dataset_name = "customer.tbl";
    ifstream in(path+dataset_name);
    if(!in)
    {
        cerr << "Cannot open the File : " << path << endl;
    }
    string line;
    string token;
    while(getline(in, line))
    {
        stringstream ss(line);
        getline(ss, token, '|');
        c_custkey.push_back(stoi(token));
        getline(ss, token, '|');
        c_name.push_back(VarChar<25>(std::wstring(token.begin(), token.end()).c_str()));
        getline(ss, token, '|');
        c_address.push_back(VarChar<40>(std::wstring(token.begin(), token.end()).c_str()));
        getline(ss, token, '|');
        c_nationkey.push_back(stoi(token));
        getline(ss, token, '|');
        c_phone.push_back(VarChar<15>(std::wstring(token.begin(), token.end()).c_str()));
        getline(ss, token, '|');
        c_acctbal.push_back(stof(token));
        getline(ss, token, '|');
        c_mktsegment.push_back(VarChar<10>(std::wstring(token.begin(), token.end()).c_str()));
        getline(ss, token, '|');
        c_comment.push_back(VarChar<117>(std::wstring(token.begin(), token.end()).c_str()));
    }

    for (int i=0; i<c_custkey.size(); i++) c_id.push_back(i);

    if (verbose)
        for(int i=0; i<5; i++)
        {
            cout << c_custkey[i] << " " << c_name[i] << " " << c_address[i] << " " << c_nationkey[i] << " " << c_phone[i] << " " << c_acctbal[i] << " " << c_mktsegment[i] << " " << c_comment[i] << endl;
        }
    
    in.close();
    cout << "Customer ... Done." << endl;
}

void load_part(string path, bool verbose=false)
{
    string dataset_name = "part.tbl";
    ifstream in(path+dataset_name);
    if(!in)
    {
        cerr << "Cannot open the File : " << path << endl;
    }
    string line;
    string token;
    while(getline(in, line))
    {
        stringstream ss(line);
        getline(ss, token, '|');
        p_partkey.push_back(stoi(token));
        getline(ss, token, '|');
        p_name.push_back(VarChar<55>(std::wstring(token.begin(), token.end()).c_str()));
        getline(ss, token, '|');
        p_mfgr.push_back(VarChar<25>(std::wstring(token.begin(), token.end()).c_str()));
        getline(ss, token, '|');
        p_brand.push_back(VarChar<10>(std::wstring(token.begin(), token.end()).c_str()));
        getline(ss, token, '|');
        p_type.push_back(VarChar<25>(std::wstring(token.begin(), token.end()).c_str()));
        getline(ss, token, '|');
        p_size.push_back(stoi(token));
        getline(ss, token, '|');
        p_container.push_back(VarChar<10>(std::wstring(token.begin(), token.end()).c_str()));
        getline(ss, token, '|');
        p_retailprice.push_back(stof(token));
        getline(ss, token, '|');
        p_comment.push_back(VarChar<23>(std::wstring(token.begin(), token.end()).c_str()));
    }

    for (int i=0; i<p_partkey.size(); i++) p_id.push_back(i);

    if (verbose)
        for(int i=0; i<5; i++)
        {
            cout << p_partkey[i] << " " << p_name[i] << " " << p_mfgr[i] << " " << p_brand[i] << " " << p_type[i] << " " << p_size[i] << " " << p_container[i] << " " << p_retailprice[i] << " " << p_comment[i] << endl;
        }

    in.close();
    cout << "Part ... Done." << endl;
}

//=======================================================================================================

int populate(string path)
{
    bool verbose = false;
    
    load_region(path, verbose);
    load_nation(path, verbose);
    load_supplier(path, verbose);
    load_part(path, verbose);
    load_customer(path, verbose);
    load_partsupp(path, verbose);
    load_orders(path, verbose);
    load_lineitem(path, verbose);
    
    cout << "=========================================" << endl;
    return 0;
}