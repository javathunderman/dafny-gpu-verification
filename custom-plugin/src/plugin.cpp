#include "clang/AST/AST.h"

#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"
#include "clang/AST/RawCommentList.h"
#include "clang/Analysis/Analyses/Dominators.h"
#include "clang/Rewrite/Core/Rewriter.h"

#include "../include/dafny.hpp"
#define DEBUG_ARR_SUBSCRIPT
#define DEBUG_VAR_LOC
using namespace clang;
using namespace clang::tooling;
static llvm::cl::OptionCategory MyToolCategory("my-tool options");
using ArgValue = std::variant<const clang::VarDecl*, const clang::IntegerLiteral*>;
std::unordered_map<std::string, std::string> ind_constraints;
std::vector<std::string> comment_reqs;
std::unordered_map<std::string, clang::Expr*> lexical_map;
std::unordered_map<std::string, clang::FunctionDecl *> kernel_map;
std::unordered_map<std::string, ArgValue> kernel_vars;
std::vector<ArraySubscriptExpr*> rewrite_ind;
class AssertionVisitor : public RecursiveASTVisitor<AssertionVisitor> {
 public:
  explicit AssertionVisitor(ASTContext *Context) : context(Context) {}
  FunctionDecl *curr_func;
  bool VisitStmt(Stmt *stmt) {
        #ifdef DEBUG_STMT
        stmt->printPretty(llvm::outs(),
                          nullptr,
                          context->getPrintingPolicy());
        #endif
        return true; 
    }
    bool VisitVarDecl(VarDecl *VD) {
        // Check if the variable is defined (not just declared)
        if (context->getSourceManager().getFileID(VD->getLocation()) == context->getSourceManager().getMainFileID() && VD->hasDefinition()) {
            if (VD->getType().getAsString() == "dim3") {
                #ifdef DEBUG_VAR_LOC
                llvm::outs() << "Variable defined: " << VD->getNameAsString() << "\n";
                llvm::outs() << "Definition located at: "
                            << VD->getLocation().printToString(context->getSourceManager())
                            << "\n";
                VD->getInit()->printPretty(llvm::outs(), nullptr, context->getPrintingPolicy());
                llvm::outs() << "\n";
                #endif
                lexical_map[VD->getNameAsString()] = VD->getInit();
            } else if (VD->getType()->isIntegerType()) {
                DeclContext *parentContext = VD->getDeclContext();
                if (clang::FunctionDecl *funcDecl = llvm::dyn_cast<clang::FunctionDecl>(parentContext)) {
                    auto lookup = kernel_map.find(funcDecl->getNameAsString());
                    if (lookup != kernel_map.end()) {
                        #ifdef DEBUG_VAR_LOC
                        llvm::outs() << "Belongs to a kernel function " << funcDecl->getNameAsString() << " " << VD->getNameAsString() << "\n";
                        #endif
                        kernel_vars[VD->getNameAsString()] = VD;
                    }
                }
            }
        }
        return true;
    }
    
    bool VisitArraySubscriptExpr(clang::ArraySubscriptExpr *arrSubExpr) {
        Expr *indexExpr = arrSubExpr->getIdx();
        Expr *baseExpr = arrSubExpr->getBase();
        auto lookup = kernel_map.find(curr_func->getNameInfo().getName().getAsString());
        if (lookup != kernel_map.end()) {
            if (clang::ImplicitCastExpr *baseExprI = llvm::dyn_cast<clang::ImplicitCastExpr>(baseExpr)) {
                #ifdef DEBUG_ARR_SUBSCRIPT
                baseExprI->getSubExprAsWritten()->printPretty(llvm::outs(),
                                nullptr,
                                context->getPrintingPolicy());
                llvm::outs() << "\n";
                #endif
                SourceRange range_base = baseExprI->getSubExprAsWritten()->getSourceRange();
                SourceManager &SM = context->getSourceManager();
                llvm::StringRef base_text_ref = Lexer::getSourceText(CharSourceRange::getTokenRange(range_base), SM, context->getLangOpts());
                std::string base_text(base_text_ref.begin(), base_text_ref.end());
                if (clang::ImplicitCastExpr *indexExprI = llvm::dyn_cast<clang::ImplicitCastExpr>(indexExpr)) {
                    #ifdef DEBUG_ARR_SUBSCRIPT
                    indexExprI->getSubExprAsWritten()->printPretty(llvm::outs(),
                                nullptr,
                                context->getPrintingPolicy());
                    llvm::outs()<< "Implicit cast expr for array subscript\n";
                    #endif
                    SourceRange range_ind = indexExprI->getSubExprAsWritten()->getSourceRange();
                    llvm::StringRef text = Lexer::getSourceText(CharSourceRange::getTokenRange(range_ind), SM, context->getLangOpts());
                    std::string stdStr(text.begin(), text.end());
                    ind_constraints[base_text] = stdStr;

                } else if (clang::BinaryOperator *indexExprB = llvm::dyn_cast<clang::BinaryOperator>(indexExpr)) {
                    #ifdef DEBUG_ARR_SUBSCRIPT
                    indexExprB->printPretty(llvm::outs(),
                                nullptr,
                                context->getPrintingPolicy());
                    llvm::outs()<< "\nBinOp for array subscript\nLHS\n";
                    indexExprB->getLHS()->printPretty(llvm::outs(), nullptr, context->getPrintingPolicy());
                    llvm::outs() << "\nRHS\n";
                    indexExprB->getRHS()->printPretty(llvm::outs(), nullptr, context->getPrintingPolicy());
                    llvm::outs() << "\n";
                    #endif
                    SourceRange range_ind = indexExprB->getSourceRange();
                    llvm::StringRef text = Lexer::getSourceText(CharSourceRange::getTokenRange(range_ind), SM, context->getLangOpts());
                    std::string stdStr(text.begin(), text.end());
                    ind_constraints[base_text] = stdStr;
                }
                llvm::outs() << "\n";
                rewrite_ind.push_back(arrSubExpr);
            }
        }
        return true;
    }
    bool VisitCallExpr(CallExpr *CE) {
        if (CE->getNumArgs() >= 2) {
            SourceManager &SM = context->getSourceManager();
            Expr *FirstArg = CE->getArg(0);
            Expr *SecondArg = CE->getArg(1);
            if (FirstArg->getType().getAsString() == "dim3" && 
                SecondArg->getType().getAsString() == "dim3") {
                #ifdef DEBUG_CALLS
                llvm::outs() << "Found CUDA kernel launch: "
                             << CE->getDirectCallee()->getNameInfo().getAsString() << "\n";
                #endif
                SourceRange range = FirstArg->getSourceRange();
                if (range.isValid()) {
                    llvm::StringRef text = Lexer::getSourceText(CharSourceRange::getTokenRange(range), SM, context->getLangOpts());
                    std::string stdStr(text.begin(), text.end());
                    if (lexical_map[stdStr]) {
                        llvm::outs() << "Grid dimensions are ";
                        lexical_map[stdStr]->printPretty(llvm::outs(),
                            nullptr,
                            context->getPrintingPolicy());
                        llvm::outs() << "\n";
                    }
                }
                SourceRange range2 = SecondArg->getSourceRange();
                if (range2.isValid()) {
                    llvm::StringRef text = Lexer::getSourceText(CharSourceRange::getTokenRange(range2), SM, context->getLangOpts());
                    std::string stdStr(text.begin(), text.end());
                    if (lexical_map[stdStr]) {
                        llvm::outs() << "Block dimensions are ";
                        lexical_map[stdStr]->printPretty(llvm::outs(),
                            nullptr,
                            context->getPrintingPolicy());
                        llvm::outs() << "\n";
                    }
                }
            }
            if (CE->getDirectCallee()) {
                std::string funcName = CE->getDirectCallee()->getNameAsString();
                auto it = kernel_map.find(funcName);
                if (it != kernel_map.end()) {
                    for (int i = 0; i < CE->getNumArgs(); i++) {
                        SourceRange range = CE->getArg(i)->getSourceRange();
                        llvm::StringRef text = Lexer::getSourceText(CharSourceRange::getTokenRange(range), SM, context->getLangOpts());
                        std::string stdStr(text.begin(), text.end());
                        llvm::outs() << "    arg: " << stdStr << "\n";
                        handleArgumentExpression(CE->getArg(i), CE->getDirectCallee()->getParamDecl(i));
                    }
                }
            }
        }
        return true;
    }
    bool VisitFunctionDecl(FunctionDecl *f) {
        curr_func = f;
        if (f->hasAttr<CUDAGlobalAttr>()) {
            kernel_map[f->getNameInfo().getName().getAsString()] = f;
            llvm::outs() << "adding global function " << f->getNameInfo().getName().getAsString();
        }
        return true;
    }

  void handleArgumentExpression(const clang::Expr *E, ParmVarDecl *p) {
    E = E->IgnoreCasts(); 
    if (const auto *DefaultArg = clang::dyn_cast<clang::CXXDefaultArgExpr>(E)) {
        E = DefaultArg->getExpr(); 
        E = E->IgnoreCasts(); 
        llvm::outs() << "  -> Argument resolved from CXXDefaultArgExpr.\n";
    }

    if (const auto *Literal = clang::dyn_cast<clang::IntegerLiteral>(E)) {
        llvm::APInt Value = Literal->getValue();
        llvm::SmallString<20> StringValue;
        bool isSigned = Literal->getType()->isSignedIntegerType();
        Value.toString(StringValue, 10, isSigned); 

        llvm::outs() << "  -> Argument is Integer Literal: " 
                     << p->getNameAsString() << " " << StringValue << "\n";
        kernel_vars[p->getNameAsString()] = Literal;
        
    } else if (const auto *DRE = clang::dyn_cast<clang::DeclRefExpr>(E)) {
        if (const auto *VarDecl = clang::dyn_cast<clang::VarDecl>(DRE->getDecl())) {
            llvm::outs() << "  -> Argument is Variable: " << p->getNameAsString() << "\n";
            kernel_vars[p->getNameAsString()] = VarDecl;
        }
        
    } else {
        llvm::outs() << "  -> Argument is Complex Expression (Type: " 
                     << E->getStmtClassName() << ")\n";
    }
}
void rewriteIfIntegerLiteral(DeclRefExpr* DRE, llvm::APInt &repl, std::string &varName, clang::Rewriter &rewrite) {
    std::string name = DRE->getNameInfo().getAsString();
    auto it = kernel_vars.find(name);

    if (it != kernel_vars.end()) {
        auto *resolveIL = std::get_if<const IntegerLiteral *>(&it->second);
        if (resolveIL) {
            repl = (*resolveIL)->getValue();
            varName = name;
            std::cout << "Found integer literal to rewrite with\n";
            rewrite.ReplaceText(DRE->getSourceRange(), std::to_string(repl.getZExtValue()));
        }
    } else {
        std::cout << "no attempted rewrite\n";
    }
}

void replaceDeclRefs(BinaryOperator* BO, llvm::APInt &repl, std::string &varName, clang::Rewriter &rewrite) {
    if (!BO) return;

    Expr* LHS = BO->getLHS()->IgnoreParenImpCasts();
    Expr* RHS = BO->getRHS()->IgnoreParenImpCasts();

    if (BinaryOperator* LHS_BO = dyn_cast<BinaryOperator>(LHS)) {
        replaceDeclRefs(LHS_BO, repl, varName, rewrite);
    } else if (DeclRefExpr* LHS_DRE = dyn_cast<DeclRefExpr>(LHS)) {
        rewriteIfIntegerLiteral(LHS_DRE, repl, varName, rewrite);
    }

    if (BinaryOperator* RHS_BO = dyn_cast<BinaryOperator>(RHS)) {
        replaceDeclRefs(RHS_BO, repl, varName, rewrite);
    } else if (DeclRefExpr* RHS_DRE = dyn_cast<DeclRefExpr>(RHS)) {
        rewriteIfIntegerLiteral(RHS_DRE, repl, varName, rewrite);
    }
}

void rewriteCollectedSubscripts() {
    clang::Rewriter rewrite;
    rewrite.setSourceMgr(context->getSourceManager(), context->getLangOpts());
    for (ArraySubscriptExpr* ASE : rewrite_ind) {
        Expr* Idx = ASE->getIdx()->IgnoreParenImpCasts();
        Expr* Base = ASE->getBase()->IgnoreParenImpCasts();
        llvm::APInt repl(32, 42);
        std::string varName;
        if (BinaryOperator* BO = dyn_cast<BinaryOperator>(Idx)) {
            replaceDeclRefs(BO, repl, varName, rewrite);
        } else if (DeclRefExpr* DRE = dyn_cast<DeclRefExpr>(Idx)) {
            rewriteIfIntegerLiteral(DRE, repl, varName, rewrite);
        }
        if (DeclRefExpr* BaseDRE = dyn_cast<DeclRefExpr>(Base)) {
            rewriteIfIntegerLiteral(BaseDRE, repl, varName, rewrite);
        }
        if (repl != NULL) {
            SourceRange range_base = Base->getSourceRange();
            SourceRange range_idx = Idx->getSourceRange();
            SourceManager &SM = context->getSourceManager();
            llvm::StringRef base_text_ref = Lexer::getSourceText(CharSourceRange::getTokenRange(range_base), SM, context->getLangOpts());
            std::string base_text(base_text_ref.begin(), base_text_ref.end());

            clang::CharSourceRange CSR = clang::CharSourceRange::getTokenRange(range_idx);

            std::string rewrittenSubscript = rewrite.getRewrittenText(CSR);
            llvm::outs() << "completed rewrite " << rewrittenSubscript << "\n";
            ind_constraints[base_text] = rewrittenSubscript;
        }
    }
    rewrite_ind.clear();
}
private:
  ASTContext *context;
};



class AssertionConsumer : public clang::ASTConsumer {
 public:
  explicit AssertionConsumer(ASTContext *Context) : visitor_(Context) {}

    virtual void HandleTranslationUnit(clang::ASTContext& context) {
        visitor_.TraverseDecl(context.getTranslationUnitDecl());
        visitor_.rewriteCollectedSubscripts();
        auto comments = context.Comments.getCommentsInFile(
            context.getSourceManager().getMainFileID());
        if (!context.Comments.empty()) {
            for (auto it = comments->begin(); it != comments->end(); it++) {
                clang::RawComment* comment = it->second;
                std::string source = comment->getFormattedText(context.getSourceManager(),
                    context.getDiagnostics());
                comment_reqs.push_back(source);
            }
        }
    }

 private:
  AssertionVisitor visitor_;
};

class AssertionFrontendAction : public clang::ASTFrontendAction {
 public:
  virtual std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(
      clang::CompilerInstance &Compiler, llvm::StringRef InFile) {
    return std::make_unique<AssertionConsumer>(&Compiler.getASTContext());
  }
};

int main(int argc, const char **argv) {
  auto ExpectedParser = CommonOptionsParser::create(argc, argv, MyToolCategory);
  if (!ExpectedParser) {
    llvm::errs() << ExpectedParser.takeError();
    return 1;
  }
  CommonOptionsParser &OptionsParser = ExpectedParser.get();

  ClangTool tool(OptionsParser.getCompilations(),
                 OptionsParser.getSourcePathList());

  AssertionFrontendAction action;
  tool.run(newFrontendActionFactory<AssertionFrontendAction>().get());
  std::string dafnyCode = readFile("test_matmul.dfy");
  std::string modifiedCode = dafnyCode;
  for (const auto& pair : ind_constraints) {
    std::cout << "writing new indexing constraint" << std::endl;
    modifiedCode = modifyDafnyCode(modifiedCode, "var idx" + pair.first + " := ", ";", pair.second, true);
  }
  for (std::string item : comment_reqs) {
    if (item.find("requires") != std::string::npos) {
      modifiedCode = modifyDafnyCode(modifiedCode, ")", "requires", item, false);
    } else if (item.find("=") != std::string::npos) {
    //   size_t pos = item.find("=");

    //   std::string prefix = "var " + item.substr(0, pos) + ":= ";
    //   std::cout << "found var to replace " << prefix << std::endl;
    //   size_t pos2 = modifiedCode.find(prefix);
    //   size_t methodEnd = modifiedCode.find(";", pos2);
    //   if (pos2 != std::string::npos) {
    //     modifiedCode.replace(pos2 + prefix.length(), methodEnd - (pos2 + prefix.length()), item.substr(pos + 1));
    //   } else {
    //     std::cout << "did not replace anything " << std::endl;
    //   }
    }
  }
  writeFile("modified_output.dfy", modifiedCode);
  std::cout << "Modified Dafny code written to: modified_output.dfy" << std::endl;
  return 0;
}